import os
import h5py
from torch.utils.data import Dataset
import numpy as np


class ArgoH5Dataset(Dataset):
    def __init__(self, dset_path, split_name="train", orig_ego=False, use_map_lanes=True):
        self.data_root = dset_path
        self.split_name = split_name
        self.orig_ego = orig_ego
        self.pred_horizon = 30
        self.num_others = 5
        self.map_attr = 2
        self.k_attr = 2
        self.use_map_lanes = use_map_lanes
        self.scene_context_img = True
        self.predict_yaw = False

        dataset = h5py.File(os.path.join(self.data_root, split_name+'_dataset.hdf5'), 'r')
        self.dset_len = len(dataset["ego_trajectories"])

    def get_input_output_seqs(self, ego_data, agents_data):
        in_len = 20

        # Ego
        ego_in = ego_data[:in_len]
        ego_out = ego_data[in_len:]

        # Other agents
        agents_in = agents_data[:in_len, :self.num_others]
        return ego_in, ego_out, agents_in

    def __getitem__(self, idx: int):
        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        ego_data = dataset['ego_trajectories'][idx]
        agents_data = dataset['agents_trajectories'][idx]
        ego_in, ego_out, agents_in = self.get_input_output_seqs(ego_data, agents_data)

        if self.use_map_lanes:
            roads = dataset['road_pts'][idx]
        else:
            roads = np.zeros((1, 1))  # dummy

        if "test" in self.split_name:
            extra = dataset['extras'][idx]
            return ego_in, agents_in, roads, extra
        elif self.orig_ego:  # for validation with re-rotation to global coordinates
            extra = dataset['extras'][idx]
            ego_data = dataset['orig_egos'][idx]
            ego_out = ego_data[20:]
            return ego_in, ego_out, agents_in, roads, extra

        return ego_in, ego_out, agents_in, roads

    def __len__(self):
        return self.dset_len


if __name__ == '__main__':
    dst = ArgoH5Dataset(dset_path="/hdd2/argoverse", split_name="train", use_map_lanes=True)
