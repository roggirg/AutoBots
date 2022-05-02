import cv2
import h5py
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import warnings
warnings.filterwarnings("ignore")


class NuscenesH5Dataset(Dataset):
    def __init__(self, dset_path, split_name="train", rtn_extras=False, model_type="Autobot-Joint",
                 use_map_img=False, use_map_lanes=True):
        self.data_root = dset_path
        self.split_name = split_name
        self.rtn_extras = rtn_extras
        self.use_map_img = use_map_img
        self.use_map_lanes = use_map_lanes
        self.pred_horizon = 12
        self.num_others = 7
        self.map_attr = 3
        self.predict_yaw = False

        if "Joint" in model_type:
            self.k_attr = 4
            self.use_joint_version = True
        else:
            self.k_attr = 2
            self.use_joint_version = False

        if self.use_joint_version and self.use_map_img:
            raise Exception("Cannot use joint version with map image. Use Map lanes instead...")

        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        self.dset_len = len(dataset["ego_trajectories"])

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.unique_agent_types = [
            ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
             'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
             'human.pedestrian.wheelchair'],
            ['vehicle.bicycle'],
            ['vehicle.motorcycle'],
            ['vehicle.car', 'vehicle.emergency.police'],
            ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
            ['vehicle.construction', 'vehicle.trailer', 'vehicle.truck']
        ]
        self.num_agent_types = len(self.unique_agent_types)

    def get_input_output_seqs(self, ego_data, agents_data):
        traj_len = int(np.sum(ego_data[:, 2]))

        # Ego
        temp_ego_in = ego_data[:traj_len - self.pred_horizon]  # at most shape(6, 3)
        ego_out = ego_data[traj_len - self.pred_horizon:traj_len]  # guaranteed to be shape(horizon, 3)
        ego_in = np.zeros((6, 3))
        ego_in[-len(temp_ego_in):] = temp_ego_in

        # separate agents into in/out.
        temp_agents_in = agents_data[:traj_len - self.pred_horizon]  # at most shape(6, 3)
        agents_in = np.zeros((6, self.num_others, 3))
        agents_in[-len(temp_agents_in):] = temp_agents_in[:, :self.num_others]
        agents_out = agents_data[traj_len - self.pred_horizon:traj_len]

        # For competition, only allowed 4 input timesteps at most.
        ego_in = ego_in[-4:]
        agents_in = agents_in[-4:]

        return ego_in, ego_out, agents_in, agents_out

    def select_valid_others(self, agents_data, agent_types):
        # Other agents: we want agents that are not parked and that are the closest to the ego agent.
        dist_travelled = []
        for n in range(agents_data.shape[1]):
            agent_traj_len = np.sum(agents_data[:, n, 2])
            idx_first_pt = np.argmax(agents_data[:, n, 2])
            dist_travelled.append(np.linalg.norm(agents_data[idx_first_pt + int(agent_traj_len) - 1, n, :2] -
                                                 agents_data[idx_first_pt, n, :2], axis=-1))

        vehicle_agents = agent_types.argmax(axis=1) >= -1
        agents_with_atleast_2_ts = agents_data[2:6, :, 2].sum(axis=0) >= 1
        active_agents = (np.array(dist_travelled) > 3.0)  # * vehicle_agents[1:] * agents_with_atleast_2_ts # 3.0
        if agent_types is not None:
            agent_types = agent_types[np.concatenate(([True], active_agents))]

        agents_data = agents_data[:, active_agents]
        if np.sum(active_agents) < self.num_others:
            temp_agents_data = np.zeros((len(agents_data), self.num_others, 3))
            temp_agents_data[:, :np.sum(active_agents)] = agents_data
            agents_data = temp_agents_data.copy()
            if agent_types is not None:
                temp_agent_types = np.zeros((self.num_others+1, agent_types.shape[1])) - 1
                temp_agent_types[:np.sum(active_agents)+1] = agent_types
                agent_types = temp_agent_types.copy()

        elif np.sum(active_agents) > self.num_others:
            # agents are already sorted from raw dataset creation.
            agents_data = agents_data[:, :self.num_others]
            if agent_types is not None:
                agent_types = agent_types[:(self.num_others+1)]

        return agents_data, agent_types

    def get_agent_types(self, agent_types_strings, num_raw_agents):
        type_to_onehot = np.eye(self.num_agent_types)
        agent_types_id = []
        for agent_type in agent_types_strings:
            for i, dset_types in enumerate(self.unique_agent_types):
                if agent_type.decode("utf-8") in dset_types:
                    agent_types_id.append(type_to_onehot[i])
        agent_types_id = np.array(agent_types_id)
        if len(agent_types_id) < num_raw_agents+1:
            new_agent_types_id = np.zeros((num_raw_agents+1, self.num_agent_types)) - 1
            new_agent_types_id[:len(agent_types_id)] = agent_types_id
            agent_types_id = new_agent_types_id.copy()

        return agent_types_id

    def mirror_scene(self, ego_in, ego_out, agents_in, agents_out, roads):
        if self.use_map_img:
            roads = cv2.flip(roads.transpose((1, 2, 0)), 1).transpose((2, 0, 1))
        elif self.use_map_lanes:
            roads[:, :, :, 0] = -roads[:, :, :, 0]
            roads[:, :, :, 2] = np.pi - roads[:, :, :, 2]
        ego_in[:, 0] = -ego_in[:, 0]
        ego_out[:, 0] = -ego_out[:, 0]
        agents_in[:, :, 0] = -agents_in[:, :, 0]
        agents_out[:, :, 0] = -agents_out[:, :, 0]
        if self.use_joint_version:
            agents_in[:, :, 2] = -agents_in[:, :, 2]
            agents_out[:, :, 2] = -agents_out[:, :, 2]
        return ego_in, ego_out, agents_in, agents_out, roads

    def make_2d_rotation_matrix(self, angle_in_radians: float) -> np.ndarray:
        """
        Makes rotation matrix to rotate point in x-y plane counterclockwise
        by angle_in_radians.
        """
        return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                         [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

    def convert_global_coords_to_local(self, coordinates: np.ndarray, yaw: float) -> np.ndarray:
        """
        Converts global coordinates to coordinates in the frame given by the rotation quaternion and
        centered at the translation vector. The rotation is meant to be a z-axis rotation.
        :param coordinates: x,y locations. array of shape [n_steps, 2].
        :param translation: Tuple of (x, y, z) location that is the center of the new frame.
        :param rotation: Tuple representation of quaternion of new frame.
            Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
        :return: x,y locations in frame stored in array of share [n_times, 2].
        """
        transform = self.make_2d_rotation_matrix(angle_in_radians=yaw)
        if len(coordinates.shape) > 2:
            coord_shape = coordinates.shape
            return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
        return np.dot(transform, coordinates.T).T[:, :2]

    def get_agent_roads(self, roads, agents_in):
        N = 100
        curr_roads = roads.copy()
        curr_roads[np.where(curr_roads[:, :, -1] == 0)] = np.nan
        mean_roads = np.nanmean(curr_roads, axis=1)[:, :2]

        # Ego Agent
        args_closest_roads = np.argsort(np.linalg.norm(np.array([[0.0, 0.0]]) - mean_roads, axis=-1))
        if len(args_closest_roads) >= N:
            per_agent_roads = [roads[args_closest_roads[:N]]]
        else:
            ego_roads = np.zeros((N, 40, 4))
            ego_roads[:len(args_closest_roads)] = roads[args_closest_roads]
            per_agent_roads = [ego_roads]

        # Other Agents
        for n in range(self.num_others):
                if agents_in[-1, n, 2]:
                    args_closest_roads = np.argsort(np.linalg.norm(agents_in[-1:, n, :2] - mean_roads, axis=-1))
                    if len(args_closest_roads) >= N:
                        per_agent_roads.append(roads[args_closest_roads[:N]])
                    else:
                        agent_roads = np.zeros((N, 40, 4))
                        agent_roads[:len(args_closest_roads)] = roads[args_closest_roads]
                        per_agent_roads.append(agent_roads)
                else:
                    per_agent_roads.append(np.zeros((N, 40, 4)))

        roads = np.array(per_agent_roads)
        roads[:, :, :, 2][np.where(roads[:, :, :, 2] < 0.0)] += 2 * np.pi  # making all orientations between 0 and 2pi

        # ensure pt closest to ego has an angle of pi/2
        temp_ego_roads = roads[0].copy()
        temp_ego_roads[np.where(temp_ego_roads[:, :, -1] == 0)] = np.nan
        dist = np.linalg.norm(temp_ego_roads[:, :, :2] - np.array([[[0.0, 0.0]]]), axis=-1)
        closest_pt = temp_ego_roads[np.where(dist == np.nanmin(dist))]
        angle_diff = closest_pt[0, 2] - (np.pi/2)
        roads[:, :, :, 2] -= angle_diff
        return roads

    def rotate_agent_datas(self, ego_in, ego_out, agents_in, agents_out, roads):
        new_ego_in = np.zeros((len(ego_in), ego_in.shape[1]+2))
        new_ego_out = np.zeros((len(ego_out), ego_out.shape[1] + 2))
        new_agents_in = np.zeros((len(agents_in), self.num_others, agents_in.shape[2]+2))
        new_agents_out = np.zeros((len(agents_out), self.num_others, agents_out.shape[2] + 2))
        new_roads = roads.copy()

        # Ego trajectories
        new_ego_in[:, :2] = ego_in[:, :2]
        new_ego_in[:, 2:] = ego_in
        new_ego_out[:, :2] = ego_out[:, :2]
        new_ego_out[:, 2:] = ego_out

        for n in range(self.num_others):
            new_agents_in[:, n, 2:] = agents_in[:, n]
            new_agents_out[:, n, 2:] = agents_out[:, n]
            if agents_in[:, n, -1].sum() >= 2:
                # then we can use the past to compute the angle to +y-axis
                diff = agents_in[-1, n, :2] - agents_in[-2, n, :2]
                yaw = np.arctan2(diff[1], diff[0])
                angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
                translation = agents_in[-1, n, :2]
            elif agents_in[:, n, -1].sum() == 1:
                # then we need to use the future to compute angle to +y-axis
                diff = agents_out[0, n, :2] - agents_in[-1, n, :2]
                yaw = np.arctan2(diff[1], diff[0])
                translation = agents_in[-1, n, :2]
                angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            else:
                # the agent does not exist...
                angle_of_rotation = None
                translation = None

            if angle_of_rotation is not None:
                new_agents_in[:, n, :2] = self.convert_global_coords_to_local(coordinates=agents_in[:, n, :2] - translation, yaw=angle_of_rotation)
                new_agents_out[:, n, :2] = self.convert_global_coords_to_local(coordinates=agents_out[:, n, :2] - translation, yaw=angle_of_rotation)
                new_agents_in[:, n, :2][np.where(new_agents_in[:, n, -1] == 0)] = 0.0
                new_agents_out[:, n, :2][np.where(new_agents_out[:, n, -1] == 0)] = 0.0
                if self.use_map_lanes:
                    new_roads[n+1, :, :, :2] = self.convert_global_coords_to_local(coordinates=new_roads[n+1, :, :, :2] - translation, yaw=angle_of_rotation)
                    new_roads[n+1, :, :, 2] -= angle_of_rotation
                    new_roads[n+1][np.where(new_roads[n+1, :, :, -1] == 0)] = 0.0

        return new_ego_in, new_ego_out, new_agents_in, new_agents_out, new_roads

    def __getitem__(self, idx: int):
        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')

        ego_data = dataset['ego_trajectories'][idx]
        agents_data = dataset['agents_trajectories'][idx]

        agent_types = self.get_agent_types(dataset['agents_types'][idx], num_raw_agents=agents_data.shape[1])
        agents_data, agent_types = self.select_valid_others(agents_data, agent_types)

        in_ego, out_ego, in_agents, out_agents = self.get_input_output_seqs(ego_data, agents_data)

        if self.use_map_img:
            # original image is 75m behind, 75m in front, 75m left, 75m right @ 0.2m/px resolution.
            # below recovers an image with 0m behind, 75m in front, 30m left, 30m right
            road_img_data = dataset['large_roads'][idx][0:375, 225:525]
            road_img_data = cv2.resize(road_img_data, dsize=(128, 128))
            roads = self.transforms(road_img_data).numpy()
        elif self.use_map_lanes:
            roads = dataset['road_pts'][idx]
            roads = self.get_agent_roads(roads, in_agents)
        else:
            roads = np.ones((1, 1))

        # Normalize all other agents futures
        # make agents have 2 sets of x,y positions (one centered @(0,0) and pointing up, and the other being raw
        if self.use_joint_version:
            in_ego, out_ego, in_agents, out_agents, roads = \
                self.rotate_agent_datas(in_ego, out_ego, in_agents, out_agents, roads)

        city_name = dataset['scene_ids'][idx][-1].decode('utf-8')
        if "train" in self.split_name:
            should_we_mirror = np.random.choice([0, 1])
            if should_we_mirror:
                in_ego, out_ego, in_agents, out_agents, roads = self.mirror_scene(in_ego, out_ego, in_agents, out_agents, roads)

        if self.use_joint_version:
            if self.rtn_extras:
                extras = [dataset['translation'][idx], dataset['rotation'][idx], dataset['scene_ids'][idx][2].decode("utf-8")]
                return in_ego, out_ego, in_agents, out_agents, roads, agent_types, extras, dataset['large_roads'][idx]

            return in_ego, out_ego, in_agents, out_agents, roads, agent_types
        else:
            if self.use_map_lanes:
                ego_roads = roads[0]
            else:
                ego_roads = roads

            if self.rtn_extras:
                # translation, rotation, instance_token, sample_token
                extras = [dataset['translation'][idx], dataset['rotation'][idx],
                          dataset['scene_ids'][idx][0].decode("utf-8"), dataset['scene_ids'][idx][1].decode("utf-8")]
                return in_ego, out_ego, in_agents, ego_roads, extras

            return in_ego, out_ego, in_agents, ego_roads

    def __len__(self):
        return self.dset_len

