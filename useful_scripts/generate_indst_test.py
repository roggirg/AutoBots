import argparse
import csv
import os
import glob
from collections import namedtuple

import pandas as pd
import xml.etree.ElementTree as xml

from datasets.interaction_dataset.utils import get_x_y_lists, get_subtype, get_type, Point, LL2XYProjector, \
    get_relation_members
from models.autobot_joint import AutoBotJoint
from process_args import load_config

import numpy as np
import torch

from utils.metric_helpers import collisions_for_inter_dataset, interpolate_trajectories, yaw_from_predictions


def load_model(model_config, models_path, device):
    autobot_model = AutoBotJoint(k_attr=8,
                                 d_k=model_config.hidden_size,
                                 _M=8,
                                 c=model_config.num_modes,
                                 T=15,
                                 L_enc=model_config.num_encoder_layers,
                                 dropout=model_config.dropout,
                                 num_heads=model_config.tx_num_heads,
                                 L_dec=model_config.num_decoder_layers,
                                 tx_hidden_size=model_config.tx_hidden_size,
                                 use_map_lanes=model_config.use_map_lanes,
                                 map_attr=7,
                                 num_agent_types=2,
                                 predict_yaw=True).to(device)

    model_dicts = torch.load(models_path, map_location=device)
    autobot_model.load_state_dict(model_dicts["AutoBot"])
    autobot_model.eval()

    model_parameters = filter(lambda p: p.requires_grad, autobot_model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Model Parameters:", num_params)

    return autobot_model


def get_dataset_files(dataset_root):
    datafiles = glob.glob(os.path.join(dataset_root, "test_multi-agent") + "/*.csv")
    return datafiles


def get_map_lanes(filename):
    print(filename)
    projector = LL2XYProjector(0.0, 0.0)

    e = xml.parse(filename).getroot()

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    unknown_linestring_types = list()
    road_lines = []

    road_lines_dict = {}
    exlusion_ids = []

    min_length = 40
    for way in e.findall('way'):
        way_type = get_type(way)

        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone":
            mark_type = np.array([1.0, 0.0, 0.0, 1.0])
        elif way_type == "line_thin":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                mark_type = np.array([1.0, 1.0, 0.0, 1.0])
            else:
                mark_type = np.array([0.0, 1.0, 0.0, 1.0])
        elif way_type == "line_thick":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                mark_type = np.array([1.0, 1.0, 0.0, 1.0])
            else:
                mark_type = np.array([0.0, 1.0, 0.0, 1.0])
        elif way_type == "pedestrian_marking":
            mark_type = np.array([0.0, 0.0, 1.0, 1.0])
        elif way_type == "bike_marking":
            mark_type = np.array([0.0, 0.0, 1.0, 1.0])
        elif way_type == "stop_line":
            mark_type = np.array([1.0, 0.0, 1.0, 1.0])
        elif way_type == "virtual":
            exlusion_ids.append(way.get("id"))
            continue
        elif way_type == "road_border":
            mark_type = np.array([1.0, 1.0, 1.0, 1.0])
        elif way_type == "guard_rail":
            mark_type = np.array([1.0, 1.0, 1.0, 1.0])
        elif way_type == "traffic_sign":
            exlusion_ids.append(way.get("id"))
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        x_list, y_list = get_x_y_lists(way, point_dict)
        if len(x_list) < min_length:
            x_list = np.linspace(x_list[0], x_list[-1], min_length).tolist()
            y_list = np.linspace(y_list[0], y_list[-1], min_length).tolist()

        lane_pts = np.array([x_list, y_list]).transpose()
        mark_type = np.zeros((len(lane_pts), 4)) + mark_type

        lane_pts = np.concatenate((lane_pts, mark_type), axis=1)
        road_lines.append(lane_pts)
        road_lines_dict[way.get("id")] = lane_pts

    new_roads = np.zeros((160, 40, 6))  # empirically found max num roads is 157.
    for i in range(len(road_lines)):
        new_roads[i, :len(road_lines[i])] = road_lines[i]

    used_keys_all = []
    num_relations = len(e.findall('relation'))
    relation_lanes = np.zeros((num_relations + len(road_lines), 80, 8))
    counter = 0
    for rel in e.findall('relation'):
        rel_lane, used_keys = get_relation_members(rel, road_lines_dict, exlusion_ids)
        if rel_lane is None:
            continue
        used_keys_all += used_keys
        new_lanes = np.array(rel_lane).reshape((-1, 8))
        relation_lanes[counter] = new_lanes
        counter += 1

    # delete all used keys
    used_keys_all = np.unique(used_keys_all)
    for used_key in used_keys_all:
        del road_lines_dict[used_key]

    # add non-used keys
    for k in road_lines_dict.keys():
        relation_lanes[counter, :40, :5] = road_lines_dict[k][:, :5]  # rest of state (position (2), and type(3)).
        relation_lanes[counter, :40, 5:7] = -1.0  # no left-right relationship
        relation_lanes[counter, :40, 7] = road_lines_dict[k][:, -1]  # mask
        counter += 1

    return relation_lanes[relation_lanes[:, :, -1].sum(1) > 0]


def get_ego_and_agents(agents_data):
    agent_masks = np.ones((*agents_data.shape[:2], 1))
    agents_data = np.concatenate((agents_data, agent_masks), axis=-1)
    ego_in = agents_data[0]
    agents_in = agents_data[1:]
    return ego_in, agents_in


def copy_agent_roads_across_agents(agents_in, roads):
    num_agents = agents_in.shape[0] + 1
    new_roads = np.zeros((num_agents, *roads.shape))
    new_roads[0] = roads  # ego
    for n in range(num_agents-1):
        if agents_in[n, -1, -1]:
            new_roads[n+1] = roads
    return new_roads


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_global_coords_to_local(coordinates: np.ndarray, yaw: float) -> np.ndarray:
        """
        Converts global coordinates to coordinates in the frame given by the rotation quaternion and
        centered at the translation vector. The rotation is meant to be a z-axis rotation.
        :param coordinates: x,y locations. array of shape [n_steps, 2].
        :param translation: Tuple of (x, y, z) location that is the center of the new frame.
        :param rotation: Tuple representation of quaternion of new frame.
            Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
        :return: x,y locations in frame stored in array of share [n_times, 2].
        """
        transform = make_2d_rotation_matrix(angle_in_radians=yaw)
        if len(coordinates.shape) > 2:
            coord_shape = coordinates.shape
            return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
        return np.dot(transform, coordinates.T).T[:, :2]


def rotate_agents(ego_in, agents_in, roads, agent_types):
    num_others = agents_in.shape[0]
    new_ego_in = np.zeros((ego_in.shape[0], ego_in.shape[1] + 3))
    new_ego_in[:, 3:] = ego_in
    new_ego_in[:, 2] = ego_in[:, 4] - ego_in[-1:, 4]  # standardized yaw.

    new_agents_in = np.zeros((agents_in.shape[0], agents_in.shape[1], agents_in.shape[2] + 3))  # + 2
    new_agents_in[:, :, 3:] = agents_in
    new_agents_in[:, :, 2] = agents_in[:, :, 4] - agents_in[:, -1:, 4]  # standardized yaw.

    new_roads = roads.copy()

    # "ego" stuff
    if agent_types[0, 0]:  # vehicle
        yaw = ego_in[-1, 4]
    elif agent_types[0, 1]:  # pedestrian/bike
        diff = ego_in[-1, :2] - ego_in[-5, :2]
        yaw = np.arctan2(diff[1], diff[0])
    angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
    translation = ego_in[-1, :2]

    new_ego_in[:, :2] = convert_global_coords_to_local(coordinates=ego_in[:, :2] - translation, yaw=angle_of_rotation)
    new_ego_in[:, 5:7] = convert_global_coords_to_local(coordinates=ego_in[:, 2:4], yaw=angle_of_rotation)
    new_roads[0, :, :, :2] = convert_global_coords_to_local(coordinates=new_roads[0, :, :, :2] - translation, yaw=angle_of_rotation)
    new_roads[0][np.where(new_roads[0, :, :, -1] == 0)] = 0.0

    # other agents
    for n in range(num_others):
        if not agents_in[n, -1, -1]:
            continue

        if agent_types[n+1, 0]:  # vehicle
            yaw = agents_in[n, -1, 4]
        elif agent_types[n+1, 1]:  # pedestrian/bike
            diff = agents_in[n, -1, :2] - agents_in[n, -5, :2]
            yaw = np.arctan2(diff[1], diff[0])
        angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
        translation = agents_in[n, -1, :2]

        new_agents_in[n, :, :2] = convert_global_coords_to_local(coordinates=agents_in[n, :, :2] - translation, yaw=angle_of_rotation)
        new_agents_in[n, :, 5:7] = convert_global_coords_to_local(coordinates=agents_in[n, :, 2:4], yaw=angle_of_rotation)
        new_roads[n + 1, :, :, :2] = convert_global_coords_to_local(coordinates=new_roads[n + 1, :, :, :2] - translation, yaw=angle_of_rotation)
        new_roads[n + 1][np.where(new_roads[n + 1, :, :, -1] == 0)] = 0.0

    return new_ego_in, new_agents_in, new_roads


def data_to_tensor(ego_in, agents_in, agent_roads, agent_types):
    return torch.from_numpy(ego_in).float().to(device).unsqueeze(0), \
           torch.from_numpy(agents_in).float().to(device).unsqueeze(0), \
           torch.from_numpy(agent_roads).float().to(device).unsqueeze(0), \
           torch.from_numpy(agent_types).float().to(device).unsqueeze(0)


def get_args():
    parser = argparse.ArgumentParser(description="AutoBot")
    parser.add_argument("--models-path", type=str, required=True, help="Load model checkpoint")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset path.")
    args = parser.parse_args()

    config, model_dirname = load_config(args.models_path)
    config = namedtuple("config", config.keys())(*config.values())
    return args, config, model_dirname


if __name__ == '__main__':
    args, config, model_dirname = get_args()
    save_dir = os.path.join(model_dirname, "autobot_submission_files")
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    autobot_model = load_model(config, args.models_path, device)

    # load dataset and process
    total_scene_collisions = []
    max_num_agents = 0
    datafiles = get_dataset_files(args.dataset_root)
    for dataf in datafiles:
        # create dataframe
        sub_file_name = os.path.join(save_dir, dataf.split("/")[-1].split(".")[0].replace("_obs", "_sub.csv"))
        headers_name = ['case_id', 'track_id', 'frame_id', 'timestamp_ms', 'track_to_predict', 'interesting_agent']
        for i in range(1, 7):
            headers_name += ['x'+str(i), 'y'+str(i), 'psi_rad'+str(i)]
        sub_file_csv = [headers_name]

        # load map
        map_fname = os.path.join(args.dataset_root, "maps", dataf.split("/")[-1].split(".")[0].replace("_obs", ".osm"))
        roads = get_map_lanes(map_fname)

        data = pd.read_csv(dataf)
        scene_ids = list(set(data['case_id'].tolist()))
        for i, scene_id in enumerate(scene_ids):
            if i % 10 == 0:
                print(scene_id, "/", len(scene_ids))
            scene_data = data.loc[data['case_id'] == scene_id]

            agent_ids = list(set(data['track_id'].tolist()))
            scene_trajectories = []
            scene_agent_types = []
            scene_agent_ids = []
            interest_agent_id = -1
            for agent_id in agent_ids:
                agent_data = scene_data.loc[scene_data['track_id'] == agent_id]
                if len(agent_data['track_to_predict'].tolist()) == 0 or not agent_data['track_to_predict'].tolist()[0]:
                    continue
                if agent_data['interesting_agent'].tolist()[0]:
                    interest_agent_id = agent_id
                scene_agent_ids.append(agent_id)
                curr_agent_type = [1.0, 0.0] if 'car' in agent_data['agent_type'].tolist()[0] else [0.0, 1.0]
                scene_agent_types.append(curr_agent_type)
                assert np.array_equal(agent_data[['frame_id']].to_numpy()[:, 0], np.arange(1, 11).astype(np.int64))
                scene_trajectories.append(agent_data[['x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']].to_numpy())

            assert interest_agent_id >= 0
            scene_trajectories = np.array(scene_trajectories)
            scene_agent_types = np.array(scene_agent_types)

            ego_in, agents_in = get_ego_and_agents(scene_trajectories)
            agent_roads = copy_agent_roads_across_agents(agents_in, roads)
            translations = np.expand_dims(np.concatenate((ego_in[-1:, :2], agents_in[:, -1, :2]), axis=0), axis=0)
            ego_in, agents_in, agent_roads = rotate_agents(ego_in, agents_in, agent_roads, scene_agent_types)

            # downsample inputs across time
            ego_in = ego_in[1::2]
            agents_in = agents_in[:, 1::2].transpose(1, 0, 2)

            # prepare inputs for model
            ego_in, agents_in, agent_roads, agent_types = data_to_tensor(ego_in, agents_in, agent_roads, scene_agent_types)
            model_ego_in = ego_in.clone()
            model_ego_in[:, :, 3:5] = 0
            model_agents_in = agents_in.clone()
            model_agents_in[:, :, :, 3:5] = 0

            # These are used for rotating the trajectories of pedestrians.
            ego_in[:, :, 0:2] = ego_in[:, :, 3:5]
            agents_in[:, :, :, 0:2] = agents_in[:, :, :, 3:5]

            # generate predictions using model'
            with torch.no_grad():
                autobot_model._M = model_agents_in.shape[2]
                pred_obs, mode_probs = autobot_model(model_ego_in, model_agents_in, agent_roads, agent_types)
            pred_obs = interpolate_trajectories(pred_obs)
            pred_obs = yaw_from_predictions(pred_obs, ego_in, agents_in)
            scene_collisions, new_preds, vehicles_only = collisions_for_inter_dataset(pred_obs.cpu().numpy(),
                                                                                      agent_types.cpu().numpy(),
                                                                                      ego_in.cpu().numpy(),
                                                                                      agents_in.cpu().numpy(),
                                                                                      translations, device=device)
            total_scene_collisions.append(scene_collisions)

            for n, agent_id in enumerate(scene_agent_ids):
                for t in range(new_preds.shape[1]):
                    row_data = [scene_id, agent_id, t+11, ((t+1)*100)+1000, 1, int(agent_id == interest_agent_id)]
                    for k in range(new_preds.shape[0]):
                        curr_pred = new_preds[k, t, 0, n, :3].tolist()
                        row_data += curr_pred

                    sub_file_csv.append(row_data)

        with open(sub_file_name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(sub_file_csv)

    print("Scene collision rate", np.mean(total_scene_collisions))



