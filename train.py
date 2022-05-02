import os

import numpy as np
import random

from datasets.argoverse.dataset import ArgoH5Dataset
from datasets.interaction_dataset.dataset import InteractionDataset
from datasets.trajnetpp.dataset import TrajNetPPDataset
from models.autobot_joint import AutoBotJoint
from process_args import get_train_args

from datasets.nuscenes.dataset import NuscenesH5Dataset
from models.autobot_ego import AutoBotEgo
import torch
import torch.distributions as D
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils.metric_helpers import min_xde_K
from utils.train_helpers import nll_loss_multimodes, nll_loss_multimodes_joint


class Trainer:
    def __init__(self, args, results_dirname):
        self.args = args
        self.results_dirname = results_dirname
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.args.seed)
        else:
            self.device = torch.device("cpu")

        self.initialize_dataloaders()
        self.initialize_model()
        self.optimiser = optim.Adam(self.autobot_model.parameters(), lr=self.args.learning_rate,
                                    eps=self.args.adam_epsilon)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=args.learning_rate_sched, gamma=0.5,
                                               verbose=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dirname, "tb_files"))
        self.smallest_minade_k = 5.0  # for computing best models
        self.smallest_minfde_k = 5.0  # for computing best models

    def initialize_dataloaders(self):
        if "Nuscenes" in self.args.dataset:
            train_dset = NuscenesH5Dataset(dset_path=self.args.dataset_path, split_name="train",
                                           model_type=self.args.model_type, use_map_img=self.args.use_map_image,
                                           use_map_lanes=self.args.use_map_lanes)
            val_dset = NuscenesH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                         model_type=self.args.model_type, use_map_img=self.args.use_map_image,
                                         use_map_lanes=self.args.use_map_lanes)

        elif "interaction-dataset" in self.args.dataset:
            train_dset = InteractionDataset(dset_path=self.args.dataset_path, split_name="train",
                                            use_map_lanes=self.args.use_map_lanes, evaluation=False)
            val_dset = InteractionDataset(dset_path=self.args.dataset_path, split_name="val",
                                          use_map_lanes=self.args.use_map_lanes, evaluation=False)

        elif "trajnet++" in self.args.dataset:
            train_dset = TrajNetPPDataset(dset_path=self.args.dataset_path, split_name="train")
            val_dset = TrajNetPPDataset(dset_path=self.args.dataset_path, split_name="val")

        elif "Argoverse" in self.args.dataset:
            train_dset = ArgoH5Dataset(dset_path=self.args.dataset_path, split_name="train",
                                       use_map_lanes=self.args.use_map_lanes)
            val_dset = ArgoH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                     use_map_lanes=self.args.use_map_lanes)

        else:
            raise NotImplementedError

        self.num_other_agents = train_dset.num_others
        self.pred_horizon = train_dset.pred_horizon
        self.k_attr = train_dset.k_attr
        self.map_attr = train_dset.map_attr
        self.predict_yaw = train_dset.predict_yaw
        if "Joint" in self.args.model_type:
            self.num_agent_types = train_dset.num_agent_types

        self.train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
        )

        print("Train dataset loaded with length", len(train_dset))
        print("Val dataset loaded with length", len(val_dset))

    def initialize_model(self):
        if "Ego" in self.args.model_type:
            self.autobot_model = AutoBotEgo(k_attr=self.k_attr,
                                            d_k=self.args.hidden_size,
                                            _M=self.num_other_agents,
                                            c=self.args.num_modes,
                                            T=self.pred_horizon,
                                            L_enc=self.args.num_encoder_layers,
                                            dropout=self.args.dropout,
                                            num_heads=self.args.tx_num_heads,
                                            L_dec=self.args.num_decoder_layers,
                                            tx_hidden_size=self.args.tx_hidden_size,
                                            use_map_img=self.args.use_map_image,
                                            use_map_lanes=self.args.use_map_lanes,
                                            map_attr=self.map_attr).to(self.device)

        elif "Joint" in self.args.model_type:
            self.autobot_model = AutoBotJoint(k_attr=self.k_attr,
                                              d_k=self.args.hidden_size,
                                              _M=self.num_other_agents,
                                              c=self.args.num_modes,
                                              T=self.pred_horizon,
                                              L_enc=self.args.num_encoder_layers,
                                              dropout=self.args.dropout,
                                              num_heads=self.args.tx_num_heads,
                                              L_dec=self.args.num_decoder_layers,
                                              tx_hidden_size=self.args.tx_hidden_size,
                                              use_map_lanes=self.args.use_map_lanes,
                                              map_attr=self.map_attr,
                                              num_agent_types=self.num_agent_types,
                                              predict_yaw=self.predict_yaw).to(self.device)
        else:
            raise NotImplementedError

    def _data_to_device(self, data):
        if "Joint" in self.args.model_type:
            ego_in, ego_out, agents_in, agents_out, context_img, agent_types = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            agents_out = agents_out.float().to(self.device)
            context_img = context_img.float().to(self.device)
            agent_types = agent_types.float().to(self.device)
            return ego_in, ego_out, agents_in, agents_out, context_img, agent_types

        elif "Ego" in self.args.model_type:
            ego_in, ego_out, agents_in, roads = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            roads = roads.float().to(self.device)
            return ego_in, ego_out, agents_in, roads

    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1).cpu().numpy()
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
        return ade_losses, fde_losses

    def _compute_marginal_errors(self, preds, ego_gt, agents_gt, agents_in):
        agent_masks = torch.cat((torch.ones((len(agents_in), 1)).to(self.device), agents_in[:, -1, :, -1]), dim=-1).view(1, 1, len(agents_in), -1)
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2).unsqueeze(0).permute(0, 2, 1, 3, 4)
        error = torch.norm(preds[:, :, :, :, :2] - agents_gt[:, :, :, :, :2], 2, dim=-1) * agent_masks
        ade_losses = np.nanmean(error.cpu().numpy(), axis=1).transpose(1, 2, 0)
        fde_losses = error[:, -1].cpu().numpy().transpose(1, 2, 0)
        return ade_losses, fde_losses

    def _compute_joint_errors(self, preds, ego_gt, agents_gt):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        agents_masks = agents_gt[:, :, :, -1]
        agents_masks[agents_masks == 0] = float('nan')
        ade_losses = []
        for k in range(self.args.num_modes):
            ade_error = (torch.norm(preds[k, :, :, :, :2].transpose(0, 1) - agents_gt[:, :, :, :2], 2, dim=-1)
                         * agents_masks).cpu().numpy()
            ade_error = np.nanmean(ade_error, axis=(1, 2))
            ade_losses.append(ade_error)
        ade_losses = np.array(ade_losses).transpose()

        fde_losses = []
        for k in range(self.args.num_modes):
            fde_error = (torch.norm(preds[k, -1, :, :, :2] - agents_gt[:, -1, :, :2], 2, dim=-1) * agents_masks[:, -1]).cpu().numpy()
            fde_error = np.nanmean(fde_error, axis=1)
            fde_losses.append(fde_error)
        fde_losses = np.array(fde_losses).transpose()

        return ade_losses, fde_losses

    def autobotego_train(self):
        steps = 0
        for epoch in range(0, self.args.num_epochs):
            print("Epoch:", epoch)
            epoch_ade_losses = []
            epoch_fde_losses = []
            epoch_mode_probs = []
            for i, data in enumerate(self.train_loader):
                ego_in, ego_out, agents_in, roads = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads)

                nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, ego_out[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.args.entropy_weight,
                                                                                   kl_weight=self.args.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)

                self.optimiser.zero_grad()
                (nll_loss + adefde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
                self.optimiser.step()

                self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)

                with torch.no_grad():
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                    epoch_ade_losses.append(ade_losses)
                    epoch_fde_losses.append(fde_losses)
                    epoch_mode_probs.append(mode_probs.detach().cpu().numpy())

                if i % 10 == 0:
                    print(i, "/", len(self.train_loader.dataset)//self.args.batch_size,
                          "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                          "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                          "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))

                steps += 1

            ade_losses = np.concatenate(epoch_ade_losses)
            fde_losses = np.concatenate(epoch_fde_losses)
            mode_probs = np.concatenate(epoch_mode_probs)

            train_minade_c = min_xde_K(ade_losses, mode_probs, K=self.args.num_modes)
            train_minade_10 = min_xde_K(ade_losses, mode_probs, K=min(self.args.num_modes, 10))
            train_minade_5 = min_xde_K(ade_losses, mode_probs, K=min(self.args.num_modes, 5))
            train_minade_1 = min_xde_K(ade_losses, mode_probs, K=1)
            train_minfde_c = min_xde_K(fde_losses, mode_probs, K=min(self.args.num_modes, 10))
            train_minfde_1 = min_xde_K(fde_losses, mode_probs, K=1)
            print("Train minADE c:", train_minade_c[0], "Train minADE 1:", train_minade_1[0], "Train minFDE c:", train_minfde_c[0])

            # Log train metrics
            self.writer.add_scalar("metrics/Train minADE_{}".format(self.args.num_modes), train_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(10), train_minade_10[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(5), train_minade_5[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(1), train_minade_1[0], epoch)
            self.writer.add_scalar("metrics/Train minFDE_{}".format(self.args.num_modes), train_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Train minFDE_{}".format(1), train_minfde_1[0], epoch)

            # update learning rate
            self.optimiser_scheduler.step()

            self.autobotego_evaluate(epoch)
            self.save_model(epoch)
            print("Best minADE c", self.smallest_minade_k, "Best minFDE c", self.smallest_minfde_k)

    def autobotego_evaluate(self, epoch):
        self.autobot_model.eval()
        with torch.no_grad():
            val_ade_losses = []
            val_fde_losses = []
            val_mode_probs = []
            for i, data in enumerate(self.val_loader):
                ego_in, ego_out, agents_in, roads = self._data_to_device(data)

                # encode observations
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads)

                ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                val_ade_losses.append(ade_losses)
                val_fde_losses.append(fde_losses)
                val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_ade_losses = np.concatenate(val_ade_losses)
            val_fde_losses = np.concatenate(val_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)
            val_minade_c = min_xde_K(val_ade_losses, val_mode_probs, K=self.args.num_modes)
            val_minade_10 = min_xde_K(val_ade_losses, val_mode_probs, K=min(self.args.num_modes, 10))
            val_minade_5 = min_xde_K(val_ade_losses, val_mode_probs, K=5)
            val_minade_1 = min_xde_K(val_ade_losses, val_mode_probs, K=1)
            val_minfde_c = min_xde_K(val_fde_losses, val_mode_probs, K=self.args.num_modes)
            val_minfde_1 = min_xde_K(val_fde_losses, val_mode_probs, K=1)

            # Log val metrics
            self.writer.add_scalar("metrics/Val minADE_{}".format(self.args.num_modes), val_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(10), val_minade_10[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(5), val_minade_5[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(1), val_minade_1[0], epoch)
            self.writer.add_scalar("metrics/Val minFDE_{}".format(self.args.num_modes), val_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Val minFDE_{}".format(1), val_minfde_1[0], epoch)

            print("minADE c:", val_minade_c[0], "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                  "minFDE c:", val_minfde_c[0], "minFDE_1:", val_minfde_1[0])
            self.autobot_model.train()
            self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0])

    def autobotjoint_train(self):
        steps = 0
        for epoch in range(0, self.args.num_epochs):
            print("Epoch:", epoch)
            epoch_marg_ade_losses = []
            epoch_marg_fde_losses = []
            epoch_marg_mode_probs = []
            epoch_scene_ade_losses = []
            epoch_scene_fde_losses = []
            epoch_mode_probs = []
            for i, data in enumerate(self.train_loader):
                ego_in, ego_out, agents_in, agents_out, map_lanes, agent_types = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, map_lanes, agent_types)

                nll_loss, kl_loss, post_entropy, adefde_loss = \
                    nll_loss_multimodes_joint(pred_obs, ego_out, agents_out, mode_probs,
                                              entropy_weight=self.args.entropy_weight,
                                              kl_weight=self.args.kl_weight,
                                              use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss,
                                              agent_types=agent_types,
                                              predict_yaw=self.predict_yaw)

                self.optimiser.zero_grad()
                (nll_loss + adefde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
                self.optimiser.step()

                self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)

                with torch.no_grad():
                    ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                    epoch_marg_ade_losses.append(ade_losses.reshape(-1, self.args.num_modes))
                    epoch_marg_fde_losses.append(fde_losses.reshape(-1, self.args.num_modes))
                    epoch_marg_mode_probs.append(
                        mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                            -1, self.args.num_modes))

                    scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out)
                    epoch_scene_ade_losses.append(scene_ade_losses)
                    epoch_scene_fde_losses.append(scene_fde_losses)
                    epoch_mode_probs.append(mode_probs.detach().cpu().numpy())

                if i % 10 == 0:
                    print(i, "/", len(self.train_loader.dataset)//self.args.batch_size,
                          "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                          "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                          "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))

                steps += 1

            epoch_marg_ade_losses = np.concatenate(epoch_marg_ade_losses)
            epoch_marg_fde_losses = np.concatenate(epoch_marg_fde_losses)
            epoch_marg_mode_probs = np.concatenate(epoch_marg_mode_probs)
            epoch_scene_ade_losses = np.concatenate(epoch_scene_ade_losses)
            epoch_scene_fde_losses = np.concatenate(epoch_scene_fde_losses)
            mode_probs = np.concatenate(epoch_mode_probs)
            train_minade_c = min_xde_K(epoch_marg_ade_losses, epoch_marg_mode_probs, K=self.args.num_modes)
            train_minfde_c = min_xde_K(epoch_marg_fde_losses, epoch_marg_mode_probs, K=self.args.num_modes)
            train_sminade_c = min_xde_K(epoch_scene_ade_losses, mode_probs, K=self.args.num_modes)
            train_sminfde_c = min_xde_K(epoch_scene_fde_losses, mode_probs, K=self.args.num_modes)
            print("Train Marg. minADE c:", train_minade_c[0], "Train Marg. minFDE c:", train_minfde_c[0], "\n",
                  "Train Scene minADE c", train_sminade_c[0], "Train Scene minFDE c", train_sminfde_c[0])

            # Log train metrics
            self.writer.add_scalar("metrics/Train Marg. minADE {}".format(self.args.num_modes), train_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Train Marg. minFDE {}".format(self.args.num_modes), train_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Train Scene minADE {}".format(self.args.num_modes), train_sminade_c[0], epoch)
            self.writer.add_scalar("metrics/Train Scene minFDE {}".format(self.args.num_modes), train_sminfde_c[0], epoch)

            self.optimiser_scheduler.step()
            self.autobotjoint_evaluate(epoch)
            self.save_model(epoch)
            print("Best Scene minADE c", self.smallest_minade_k, "Best Scene minFDE c", self.smallest_minfde_k)

    def autobotjoint_evaluate(self, epoch):
        self.autobot_model.eval()
        with torch.no_grad():
            val_marg_ade_losses = []
            val_marg_fde_losses = []
            val_marg_mode_probs = []
            val_scene_ade_losses = []
            val_scene_fde_losses = []
            val_mode_probs = []
            for i, data in enumerate(self.val_loader):
                ego_in, ego_out, agents_in, agents_out, context_img, agent_types = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, context_img, agent_types)

                # Marginal metrics
                ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                val_marg_ade_losses.append(ade_losses.reshape(-1, self.args.num_modes))
                val_marg_fde_losses.append(fde_losses.reshape(-1, self.args.num_modes))
                val_marg_mode_probs.append(
                    mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                        -1, self.args.num_modes))

                # Joint metrics
                scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out)
                val_scene_ade_losses.append(scene_ade_losses)
                val_scene_fde_losses.append(scene_fde_losses)
                val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_marg_ade_losses = np.concatenate(val_marg_ade_losses)
            val_marg_fde_losses = np.concatenate(val_marg_fde_losses)
            val_marg_mode_probs = np.concatenate(val_marg_mode_probs)

            val_scene_ade_losses = np.concatenate(val_scene_ade_losses)
            val_scene_fde_losses = np.concatenate(val_scene_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)

            val_minade_c = min_xde_K(val_marg_ade_losses, val_marg_mode_probs, K=self.args.num_modes)
            val_minfde_c = min_xde_K(val_marg_fde_losses, val_marg_mode_probs, K=self.args.num_modes)
            val_sminade_c = min_xde_K(val_scene_ade_losses, val_mode_probs, K=self.args.num_modes)
            val_sminfde_c = min_xde_K(val_scene_fde_losses, val_mode_probs, K=self.args.num_modes)

            # Log train metrics
            self.writer.add_scalar("metrics/Val Marg. minADE {}".format(self.args.num_modes), val_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Val Marg. minFDE {}".format(self.args.num_modes), val_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Val Scene minADE {}".format(self.args.num_modes), val_sminade_c[0], epoch)
            self.writer.add_scalar("metrics/Val Scene minFDE {}".format(self.args.num_modes), val_sminfde_c[0], epoch)

            print("Marg. minADE c:", val_minade_c[0], "Marg. minFDE c:", val_minfde_c[0])
            print("Scene minADE c:", val_sminade_c[0], "Scene minFDE c:", val_sminfde_c[0])

            self.autobot_model.train()
            self.save_model(minade_k=val_sminade_c[0], minfde_k=val_sminfde_c[0])

    def save_model(self, epoch=None, minade_k=None, minfde_k=None):
        if epoch is None:
            if minade_k < self.smallest_minade_k:
                self.smallest_minade_k = minade_k
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_ade.pth"),
                )

            if minfde_k < self.smallest_minfde_k:
                self.smallest_minfde_k = minfde_k
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_fde.pth"),
                )

        else:
            if epoch % 10 == 0 and epoch > 0:
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "models_%d.pth" % epoch),
                )

    def train(self):
        if "Ego" in self.args.model_type:
            self.autobotego_train()
        elif "Joint" in self.args.model_type:
            self.autobotjoint_train()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    args, results_dirname = get_train_args()
    trainer = Trainer(args, results_dirname)
    trainer.train()
