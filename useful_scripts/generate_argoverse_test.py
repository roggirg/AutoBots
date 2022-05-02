from typing import Tuple

import numpy as np
from argoverse.evaluation.competition_util import generate_forecasting_h5
import torch
from sklearn.cluster import AgglomerativeClustering

from datasets.argoverse.dataset import ArgoH5Dataset
from models.autobot_ego import AutoBotEgo
from process_args import get_eval_args


def load_model(args, config, k_attr, num_other_agents, pred_horizon, map_attr):
    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    autobot_model = AutoBotEgo(k_attr=k_attr,
                               d_k=config['hidden_size'],
                               _M=num_other_agents,
                               c=config['num_modes'],
                               T=pred_horizon,
                               L_enc=config['num_encoder_layers'],
                               dropout=config['dropout'],
                               num_heads=config['tx_num_heads'],
                               L_dec=config['num_decoder_layers'],
                               tx_hidden_size=config['tx_hidden_size'],
                               use_map_img=config['use_map_image'],
                               use_map_lanes=config['use_map_lanes'],
                               map_attr=map_attr).to(device)

    model_dicts = torch.load(args.models_path, map_location={'cuda:1': 'cuda:0'})
    autobot_model.load_state_dict(model_dicts["AutoBot"])
    autobot_model.eval()

    return autobot_model, device


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_local_coords_to_global(coordinates: np.ndarray, translation: Tuple[float, float], yaw: float) -> np.ndarray:
    """
    Converts local coordinates to global coordinates.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(yaw)
    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)

    return np.dot(transform, coordinates.T).T[:, :2] + np.atleast_2d(np.array(translation)[:2])


def recompute_probs(pred_trajs, probs):
    distances = []
    for k in range(len(pred_trajs)):
        distances.append(np.mean(np.linalg.norm(pred_trajs[k] - pred_trajs, axis=-1), axis=-1))
    distances = np.array(distances)

    agg = AgglomerativeClustering(affinity='precomputed', linkage='complete', distance_threshold=None, n_clusters=4)
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
    args, config, model_dirname = get_eval_args()
    test_argoverse = ArgoH5Dataset(args.dataset_path, split_name="test", use_map_lanes=config['use_map_lanes'])
    test_loader = torch.utils.data.DataLoader(
        test_argoverse, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=False, pin_memory=False
    )
    print("Test dataset loaded with length", len(test_argoverse))

    autobot_model, device = load_model(args, config, num_other_agents=test_argoverse.num_others,
                                       pred_horizon=test_argoverse.pred_horizon, k_attr=test_argoverse.k_attr,
                                       map_attr=test_argoverse.map_attr)

    trajectories = {}
    probabilities = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 25 == 0:
                print(i, "/", len(test_argoverse) // args.batch_size)

            ego_in, agents_in, roads, extra = data
            ego_in = ego_in.float().to(device)
            agents_in = agents_in.float().to(device)
            roads = roads.float().to(device)

            pred_obs, mode_probs = autobot_model(ego_in, agents_in, roads)
            pred_obs = pred_obs.cpu().numpy()
            mode_probs = mode_probs.cpu().numpy()

            for b in range(len(mode_probs)):
                glob_pred_obs = []
                for k in range(len(pred_obs)):
                    glob_pred_obs.append(convert_local_coords_to_global(pred_obs[k, :, b, :2], extra[b, 2:].numpy(), extra[b, 1].item()))
                glob_pred_obs = np.array(glob_pred_obs)
                new_probs = recompute_probs(glob_pred_obs, mode_probs[b])
                trajectories[int(extra[b, 0].item())] = glob_pred_obs
                probabilities[int(extra[b, 0].item())] = new_probs

        fname = args.models_path.split("/")[-1].split(".")[0]
        generate_forecasting_h5(data=trajectories, output_path=model_dirname, probabilities=probabilities, filename=fname)
