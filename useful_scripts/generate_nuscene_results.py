import json
import numpy as np
import torch
from nuscenes.prediction import convert_local_coords_to_global
from sklearn.cluster import AgglomerativeClustering

from datasets.nuscenes.dataset import NuscenesH5Dataset
from models.autobot_ego import AutoBotEgo
from process_args import get_eval_args


def load_model(args, model_config, k_attr, num_other_agents, pred_horizon, map_attr):
    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    autobot_model = AutoBotEgo(k_attr=k_attr,
                               d_k=model_config.hidden_size,
                               _M=num_other_agents,
                               c=model_config.num_modes,
                               T=pred_horizon,
                               L_enc=model_config.num_encoder_layers,
                               dropout=model_config.dropout,
                               num_heads=model_config.tx_num_heads,
                               L_dec=model_config.num_decoder_layers,
                               tx_hidden_size=model_config.tx_hidden_size,
                               use_map_img=model_config.use_map_image,
                               use_map_lanes=model_config.use_map_lanes,
                               map_attr=map_attr).to(device)

    model_dicts = torch.load(args.models_path, map_location={'cuda:1': 'cuda:0'})
    autobot_model.load_state_dict(model_dicts["AutoBot"])
    autobot_model.eval()

    return autobot_model, device


def recompute_probs(pred_trajs, probs):
    distances = []
    for k in range(len(pred_trajs)):
        distances.append(np.mean(np.linalg.norm(pred_trajs[k] - pred_trajs, axis=-1), axis=-1))
    distances = np.array(distances)

    agg = AgglomerativeClustering(affinity='precomputed', linkage='complete', distance_threshold=None, n_clusters=6)
    output = agg.fit_predict(distances)  # Returns class labels.
    temp_probs = probs.copy()
    for element in np.unique(output):
        similar_ks = np.where(output == element)[0].tolist()
        if len(similar_ks) > 1:
            best_k = similar_ks[np.argmax(temp_probs[similar_ks])]
            similar_ks.remove(best_k)
            for similar_k in similar_ks:
                temp_probs[best_k] += temp_probs[similar_k]
                temp_probs[similar_k] = 0.0

    return temp_probs


if __name__ == "__main__":
    args, model_config, model_dirname = get_eval_args()

    val_dset = NuscenesH5Dataset(dset_path=args.dataset_path, split_name="val",
                                 model_type=model_config.model_type, use_map_img=model_config.use_map_image,
                                 use_map_lanes=model_config.use_map_lanes, rtn_extras=True)

    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=False, pin_memory=False
    )
    print("Val dataset loaded with length", len(val_dset))

    autobot_model, device = load_model(args, model_config, val_dset.k_attr, val_dset.num_others, val_dset.pred_horizon,
                                       val_dset.map_attr)

    preds = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i % 25 == 0:
                print(i, "/", len(val_dset) // args.batch_size)

            ego_in, ego_out, agents_in, roads, extras = data
            ego_in = ego_in.float().to(device)
            ego_out = ego_out.float().to(device)
            agents_in = agents_in.float().to(device)
            roads = roads.float().to(device)

            pred_obs, mode_preds = autobot_model(ego_in, agents_in, roads)
            pred_seqs = pred_obs[:, :, :, :2].cpu().numpy().transpose((2, 0, 1, 3))
            mode_preds = mode_preds.cpu().numpy()

            # Process extras
            translation = extras[0].cpu().numpy()
            rotation = extras[1].cpu().numpy()
            instance_tokens = list(extras[2])
            sample_tokens = list(extras[3])

            for b in range(min(args.batch_size, len(pred_seqs))):
                curr_out = {}
                curr_out["instance"] = instance_tokens[b]
                curr_out["sample"] = sample_tokens[b]
                mode_preds[b] = recompute_probs(pred_seqs[b], mode_preds[b])
                curr_out["probabilities"] = mode_preds[b].tolist()
                curr_out["prediction"] = []
                for k in range(model_config.num_modes):
                    pred_seq = pred_seqs[b, k]
                    curr_out["prediction"].append(
                        convert_local_coords_to_global(pred_seq, translation[b], rotation[b]).tolist()
                    )

                preds.append(curr_out)

        with open(model_dirname + '/autobot_preds.json', 'w') as fout:
            json.dump(preds, fout)


