import argparse
import h5py
import glob
import os
import pandas as pd
import numpy as np
from datasets.interaction_dataset.utils import get_minmax_mapfile


def get_args():
    parser = argparse.ArgumentParser(description="Interaction-Dataset H5 Creator")
    parser.add_argument("--output-h5-path", type=str, required=True, help="output path to H5 files.")
    parser.add_argument("--raw-dataset-path", type=str, required=True, help="raw Dataset path to multi-agent folder.")
    parser.add_argument("--split-name", type=str, default="train", help="split-name to create", choices=["train", "val"])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    datafiles = glob.glob(args.raw_dataset_path + args.split_name + "/*.csv")
    num_scenes = 0
    for datafile in datafiles:
        data = pd.read_csv(datafile)
        num_scenes += len(list(set(data['case_id'].tolist())))

    max_num_agents = 50  # found by inspection
    f = h5py.File(os.path.join(args.output_h5_path, args.split_name + '_dataset.hdf5'), 'w')
    agent_trajectories = f.create_dataset("agents_trajectories", shape=(num_scenes, max_num_agents, 40, 7), chunks=(1, max_num_agents, 40, 7), dtype=np.float32)
    agent_types = f.create_dataset("agents_types", shape=(num_scenes, max_num_agents, 2), chunks=(1, max_num_agents, 2), dtype=np.float32)
    metas = f.create_dataset("metas", shape=(num_scenes, 5), chunks=(1, 5))
    map_paths = f.create_dataset("map_paths", shape=(num_scenes, 1), chunks=(1, 1), dtype='S200')

    global_scene_id = 0
    for datafile_id, datafile in enumerate(datafiles):
        data = pd.read_csv(datafile)
        current_roads = datafile.split("/")[-1].split(".")[0].replace("_" + args.split_name, "")
        lanelet_map_file = os.path.join(args.raw_dataset_path, "maps", current_roads + ".osm")
        xmin, ymin, xmax, ymax = get_minmax_mapfile(lanelet_map_file)

        scene_ids = list(set(data['case_id'].tolist()))
        for i, scene_id in enumerate(scene_ids):
            if i % 10 == 0:
                print(scene_id, "/", len(scene_ids))
            scene_data = data.loc[data['case_id'] == scene_id]

            agent_ids = list(set(data['track_id'].tolist()))
            scene_trajectories = []
            scene_agent_types = []
            for agent_id in agent_ids:
                agent_data = scene_data.loc[scene_data['track_id'] == agent_id]
                if len(agent_data['frame_id'].tolist()) < 40:  # only take complete trajectories
                    continue
                curr_agent_type = [1.0, 0.0] if 'car' in agent_data['agent_type'].tolist()[0] else [0.0, 1.0]
                scene_agent_types.append(curr_agent_type)
                scene_trajectories.append(agent_data[['x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']].to_numpy())

            scene_trajectories = np.array(scene_trajectories)
            scene_agent_types = np.array(scene_agent_types)
            if len(scene_trajectories) < max_num_agents:
                temp_scene_trajectories = np.zeros((max_num_agents, 40, 7)) - 1
                temp_scene_trajectories[:len(scene_trajectories)] = scene_trajectories
                scene_trajectories = temp_scene_trajectories.copy()
                temp_scene_types = np.zeros((max_num_agents, 2)) - 1
                temp_scene_types[:len(scene_agent_types)] = scene_agent_types
                scene_agent_types = temp_scene_types.copy()

            agent_trajectories[global_scene_id] = scene_trajectories
            agent_types[global_scene_id] = scene_agent_types
            metas[global_scene_id] = np.array([xmin, ymin, xmax, ymax, datafile_id])
            map_paths[global_scene_id] = lanelet_map_file.encode("ascii", "ignore")
            global_scene_id += 1


