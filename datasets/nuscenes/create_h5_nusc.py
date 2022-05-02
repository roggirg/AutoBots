import argparse

import h5py
import os
import numpy as np

from datasets.nuscenes.raw_dataset import NuScenesDataset


'''
Train H5 generation takes about 3 hours and is about 56GBs.
Val H5 generation takes about 1 hour and is about 16GBs.
'''


def get_args():
    parser = argparse.ArgumentParser(description="Nuscenes H5 Creator")
    parser.add_argument("--output-h5-path", type=str, required=True, help="output path to H5 files.")
    parser.add_argument("--raw-dataset-path", type=str, required=True, help="raw Dataset path to v1.0-trainval_full.")
    parser.add_argument("--split-name", type=str, default="train", help="split-name to create", choices=["train", "val"])
    parser.add_argument("--ego-range", type=int, nargs="+", default=[75, 75, 75, 75],
                        help="range around ego in meters [left, right, behind, front].")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    max_num_agents = 20

    save_dir = os.path.join(args.output_h5_path, args.split_name)
    nuscenes = NuScenesDataset(data_root=args.raw_dataset_path, split_name=args.split_name,
                               version='v1.0-trainval', ego_range=args.ego_range, num_others=max_num_agents)
    num_scenes = len(nuscenes)

    f = h5py.File(os.path.join(args.output_h5_path, args.split_name + '_dataset.hdf5'), 'w')
    ego_trajectories = f.create_dataset("ego_trajectories", shape=(num_scenes, 18, 3), chunks=(1, 18, 3), dtype=np.float32)
    agent_trajectories = f.create_dataset("agents_trajectories", shape=(num_scenes, 18, max_num_agents, 3), chunks=(1, 18, max_num_agents, 3), dtype=np.float32)
    scene_ids = f.create_dataset("scene_ids", shape=(num_scenes, 3), chunks=(1, 3), dtype='S50')
    scene_translation = f.create_dataset("translation", shape=(num_scenes, 3), chunks=(1, 3))
    scene_rotation = f.create_dataset("rotation", shape=(num_scenes, 4), chunks=(1, 4))
    agent_types = f.create_dataset("agents_types", shape=(num_scenes, max_num_agents+1), chunks=(1, max_num_agents+1), dtype='S50')
    road_pts = f.create_dataset("road_pts", shape=(num_scenes, 100, 40, 4), chunks=(1, 100, 40, 4), dtype=np.float16)
    road_imgs = f.create_dataset("large_roads", shape=(num_scenes, 750, 750, 3), chunks=(1, 750, 750, 3), dtype=np.uint8)

    for i, data in enumerate(nuscenes):
        if i % 10 == 0:
            print(i, "/", num_scenes)
        ego_trajectories[i] = data[0]
        agent_trajectories[i] = data[1]

        road_imgs[i] = data[2]

        curr_scene_id = [n.encode("ascii", "ignore") for n in [data[3][0], data[3][1], data[3][4]]]
        scene_ids[i] = curr_scene_id

        scene_translation[i] = data[3][2]
        scene_rotation[i] = data[3][3]

        curr_agent_types = data[4]
        while len(curr_agent_types) < max_num_agents + 1:
            curr_agent_types.append("None")
        agent_types_ascii = [n.encode("ascii", "ignore") for n in curr_agent_types]
        agent_types[i] = agent_types_ascii

        road_pts[i] = data[5]

