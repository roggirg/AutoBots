import argparse

import h5py
import os
import time

import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from typing import Tuple


'''
Data should be extracted from  .tar.gz files into "{split_name}/data", e.g. "{args.raw_dataset_path}/train/data/*.csv"
'''


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


def convert_global_coords_to_local(coordinates: np.ndarray, translation: Tuple[float, float], yaw: float) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param yaw: heading angle of agent.
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """

    yaw = angle_of_rotation(yaw)
    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation))).T

    return np.dot(transform, coords).T[:, :2]


def compute_yaw(ego_input):
    diff = ego_input[-1] - ego_input[-10]
    return np.arctan2(diff[1], diff[0])


def get_args():
    parser = argparse.ArgumentParser(description="Argoverse H5 Creator")
    parser.add_argument("--output-h5-path", type=str, required=True, help="output path to H5 files.")
    parser.add_argument("--raw-dataset-path", type=str, required=True, help="raw Dataset path to root of extracted files")
    parser.add_argument("--split-name", type=str, default="train", help="split-name to create", choices=["train", "val", "test"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root_dir = os.path.join(args.raw_dataset_path, args.split_name, 'data')
    avm = ArgoverseMap()
    afl = ArgoverseForecastingLoader(root_dir)  # simply change to your local path of the data
    start_time = time.time()

    seq_files = sorted(os.listdir(root_dir))
    num_scenes = len(seq_files)
    print("Number of files:", num_scenes)

    if "test" in args.split_name:
        num_timesteps = 20
    else:
        num_timesteps = 50

    max_num_roads = 150  # manually found
    max_num_agents = 15  # manually found

    f = h5py.File(os.path.join(args.output_h5_path, args.split_name + '_dataset.hdf5'), 'w')
    ego_trajectories = f.create_dataset("ego_trajectories", shape=(num_scenes, num_timesteps, 3),
                                        chunks=(1, num_timesteps, 3), dtype=np.float32)
    agent_trajectories = f.create_dataset("agents_trajectories", shape=(num_scenes, num_timesteps, max_num_agents, 3),
                                          chunks=(1, num_timesteps, max_num_agents, 3), dtype=np.float32)
    road_pts = f.create_dataset("road_pts", shape=(num_scenes, max_num_roads, 10, 3),
                                chunks=(1, max_num_roads, 10, 3), dtype=np.float16)
    extras = f.create_dataset("extras", shape=(num_scenes, 4), chunks=(1, 4), dtype=np.float32)
    orig_egos = f.create_dataset("orig_egos", shape=(num_scenes, num_timesteps, 3), chunks=(1, num_timesteps, 3),
                                 dtype=np.float32)

    for i, seq_file in enumerate(seq_files):
        if i % 1000 == 0:
            print(i)
        fname = os.path.join(root_dir, seq_file)
        df = afl.get(fname).seq_df

        # Gather agents' trajectories
        ego_x = df[df["OBJECT_TYPE"] == "AGENT"]["X"]
        ego_y = df[df["OBJECT_TYPE"] == "AGENT"]["Y"]
        ego_traj = np.column_stack((ego_x, ego_y, np.ones_like(ego_x)))
        ego_timestamps = df[df["OBJECT_TYPE"] == "AGENT"]["TIMESTAMP"].values

        ego_pred_timestamp = ego_timestamps[19]
        others_traj = []
        frames = df.groupby("TRACK_ID")
        for group_name, group_data in frames:
            object_type = group_data["OBJECT_TYPE"].values[0]
            if "AGENT" in object_type:  # skip ego agent trajectory
                continue

            other_timestamp = group_data["TIMESTAMP"].values
            if ego_pred_timestamp not in other_timestamp:
                # ignore agents that are not there at the prediction time.
                continue

            other_x = group_data["X"].values
            other_y = group_data["Y"].values
            other_traj = np.column_stack((other_x, other_y, np.ones(len(other_x))))
            if 10 <= len(other_traj) < num_timesteps:  # if we have an imcomplete trajectory of an 'other' agent.
                temp_other_traj = np.zeros((num_timesteps, 3))

                for j, timestamp in enumerate(ego_timestamps):
                    if timestamp == other_timestamp[0]:
                        temp_other_traj[j:j+len(other_traj)] = other_traj
                        break
                other_traj = temp_other_traj.copy()
            elif len(other_traj) < 10:
                continue

            if np.linalg.norm(other_traj[19, :2] - ego_traj[19, :2]) > 40:
                continue

            others_traj.append(other_traj)

        # Rotating trajectories so that ego is going up.
        ego_yaw = compute_yaw(ego_traj[:20, :2])
        rot_ego_traj = convert_global_coords_to_local(ego_traj[:, :2], ego_traj[19, :2], ego_yaw)
        rot_ego_traj = np.concatenate((rot_ego_traj, ego_traj[:, 2:]), axis=-1)
        rot_others_traj = []
        for other_traj in others_traj:
            rot_other_traj = convert_global_coords_to_local(other_traj[:, :2], ego_traj[19, :2], ego_yaw)
            rot_other_traj = rot_other_traj * other_traj[:, 2:]
            rot_other_traj = np.column_stack((rot_other_traj[:, 0], rot_other_traj[:, 1], other_traj[:, 2]))
            rot_others_traj.append(rot_other_traj)
        rot_others_traj = np.array(rot_others_traj)
        if len(rot_others_traj) == 0:
            rot_others_traj = np.zeros((max_num_agents, num_timesteps, 3))
        elif len(rot_others_traj) <= max_num_agents:
            temp_rot_others_traj = np.zeros((max_num_agents, num_timesteps, 3))
            temp_rot_others_traj[:len(rot_others_traj)] = rot_others_traj
            rot_others_traj = temp_rot_others_traj.copy()
        else:
            dists = np.linalg.norm(rot_others_traj[:, 19, :2], axis=-1)
            closest_inds = np.argsort(dists)[:max_num_agents]
            rot_others_traj = rot_others_traj[closest_inds]

        # Get lane centerlines which lie within the range of trajectories and rotate all lanes
        city_name = df["CITY_NAME"].values[0]
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]
        query_bbox = (ego_traj[19, 0] - 50, ego_traj[19, 0] + 50, ego_traj[19, 1] - 50, ego_traj[19, 1] + 50)
        lane_centerlines = []
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if len(lane_cl) == 1:
                continue

            if (np.min(lane_cl[:, 0]) < query_bbox[1] and np.min(lane_cl[:, 1]) < query_bbox[3]
                    and np.max(lane_cl[:, 0]) > query_bbox[0] and np.max(lane_cl[:, 1]) > query_bbox[2]):
                lane_cl = convert_global_coords_to_local(lane_cl[:, :2], ego_traj[19, :2], ego_yaw)
                lane_cl = np.concatenate((lane_cl, np.ones((len(lane_cl), 1))), axis=-1)  # adding existence mask
                if len(lane_cl) < 10:
                    temp_lane_cl = np.zeros((10, 3))
                    temp_lane_cl[:len(lane_cl)] = lane_cl
                    lane_cl = temp_lane_cl.copy()
                elif len(lane_cl) > 10:
                    print("found lane with more than 10 pts...")
                    continue

                lane_centerlines.append(lane_cl)

        lane_centerlines = np.array(lane_centerlines)
        if len(lane_centerlines) <= max_num_roads:
            temp_lane_centerlines = np.zeros((max_num_roads, 10, 3))
            temp_lane_centerlines[:len(lane_centerlines)] = lane_centerlines
            lane_centerlines = temp_lane_centerlines.copy()

        extra = [int(seq_file.split(".")[0]), ego_yaw, ego_traj[19, 0], ego_traj[19, 1]]

        ego_trajectories[i] = rot_ego_traj
        agent_trajectories[i] = rot_others_traj.transpose(1, 0, 2)
        road_pts[i] = lane_centerlines
        orig_egos[i] = ego_traj
        extras[i] = np.array(extra)

        if (i+1) % 100 == 0:
            print("Time taken", time.time() - start_time, "Number of examples", i+1)
