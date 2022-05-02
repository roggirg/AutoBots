import xml.etree.ElementTree as xml
import h5py
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from torch.utils.data import Dataset
import numpy as np
from datasets.interaction_dataset.utils import LL2XYProjector, Point, get_type, get_subtype, get_x_y_lists, \
    get_relation_members


class InteractionDataset(Dataset):
    def __init__(self, dset_path, split_name="train", evaluation=False, use_map_lanes=True):
        self.data_root = dset_path
        self.split_name = split_name
        self.pred_horizon = 15
        self.num_agent_types = 2
        self.use_map_lanes = use_map_lanes
        self.predict_yaw = True
        self.map_attr = 7
        self.k_attr = 8

        dataset = h5py.File(os.path.join(self.data_root, split_name + '_dataset.hdf5'), 'r')
        self.dset_len = len(dataset["agents_trajectories"])

        roads_fnames = glob.glob(os.path.join(self.data_root, "maps", "*.osm"))
        self.max_num_pts_per_road_seg = 0
        self.max_num_road_segs = 0
        self.roads = {}
        for osm_fname in roads_fnames:
            road_info = self.get_map_lanes(osm_fname, 0., 0.)
            if len(road_info) > self.max_num_road_segs:
                self.max_num_road_segs = len(road_info)
            key_fname = osm_fname.split("/")[-1]
            self.roads[key_fname] = road_info

        self.max_num_agents = 0
        self.evaluation = evaluation
        if not evaluation:
            self.num_others = 8  # train with 8 agents
        else:
            self.num_others = 40  # evaluate with 40 agents.

    def get_map_lanes(self, filename, lat_origin, lon_origin):
        projector = LL2XYProjector(lat_origin, lon_origin)

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
                # mark_type = np.array([1.0, 1.0, 0.0, 1.0])
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
            if len(x_list) > self.max_num_pts_per_road_seg:
                self.max_num_pts_per_road_seg = len(x_list)

            lane_pts = np.concatenate((lane_pts, mark_type), axis=1)
            road_lines.append(lane_pts)
            road_lines_dict[way.get("id")] = lane_pts

        new_roads = np.zeros((160, self.max_num_pts_per_road_seg, 6))  # empirically found max num roads is 157.
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

    def split_input_output_normalize(self, agents_data, meta_data, agent_types):
        if self.evaluation:
            in_horizon = 10
        else:
            in_horizon = 5

        agent_masks = np.expand_dims(agents_data[:, :, 0] != -1, axis=-1).astype(np.float32)  # existence mask
        agents_data[:, :, :2] -= np.array([[meta_data[0], meta_data[1]]])  # Normalize with translation only
        agents_data = np.nan_to_num(agents_data, nan=-1.0)  # pedestrians have nans instead of yaw and size
        agents_data = np.concatenate([agents_data, agent_masks], axis=-1)

        dists = euclidean_distances(agents_data[:, in_horizon - 1, :2], agents_data[:, in_horizon - 1, :2])
        agent_masks[agent_masks == 0] = np.nan
        dists *= agent_masks[:, in_horizon - 1]
        dists *= agent_masks[:, in_horizon - 1].transpose()
        ego_idx = np.random.randint(0, int(np.nansum(agent_masks[:, in_horizon - 1])))
        closest_agents = np.argsort(dists[ego_idx])
        agents_data = agents_data[closest_agents[:self.num_others + 1]]
        agent_types = agent_types[closest_agents[:self.num_others + 1]]

        agents_in = agents_data[1:(self.num_others + 1), :in_horizon]
        agents_out = agents_data[1:(self.num_others + 1), in_horizon:, [0, 1, 4, 7]]  # returning positions and yaws
        ego_in = agents_data[0, :in_horizon]
        ego_out = agents_data[0, in_horizon:]
        ego_out = ego_out[:, [0, 1, 4, 7]]  # returning positions and yaws

        return ego_in, ego_out, agents_in, agents_out, agent_types

    def copy_agent_roads_across_agents(self, agents_in, roads):
        new_roads = np.zeros((self.num_others + 1, *roads.shape))
        new_roads[0] = roads  # ego
        for n in range(self.num_others):
            if agents_in[n, -1, -1]:
                new_roads[n + 1] = roads
        return new_roads

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

    def rotate_agents(self, ego_in, ego_out, agents_in, agents_out, roads, agent_types):
        new_ego_in = np.zeros(
            (ego_in.shape[0], ego_in.shape[1] + 3))  # +2 adding three dimensions for original positions and yaw
        new_ego_in[:, 3:] = ego_in
        new_ego_in[:, 2] = ego_in[:, 4] - ego_in[-1:, 4]  # standardized yaw.

        new_ego_out = np.zeros((ego_out.shape[0], ego_out.shape[1] + 2))  # adding two dimensions for original positions
        new_ego_out[:, 2:] = ego_out
        new_ego_out[:, 4] -= ego_in[-1:, 4]  # standardized yaw.

        new_agents_in = np.zeros((agents_in.shape[0], agents_in.shape[1], agents_in.shape[2] + 3))  # + 2
        new_agents_in[:, :, 3:] = agents_in
        new_agents_in[:, :, 2] = agents_in[:, :, 4] - agents_in[:, -1:, 4]  # standardized yaw.

        new_agents_out = np.zeros((agents_out.shape[0], agents_out.shape[1], agents_out.shape[2] + 2))
        new_agents_out[:, :, 2:] = agents_out
        new_agents_out[:, :, 4] -= agents_in[:, -1:, 4]

        new_roads = roads.copy()

        # "ego" stuff
        if agent_types[0, 0]:  # vehicle
            yaw = ego_in[-1, 4]
        elif agent_types[0, 1]:  # pedestrian/bike
            diff = ego_in[-1, :2] - ego_in[-5, :2]
            yaw = np.arctan2(diff[1], diff[0])
        angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
        translation = ego_in[-1, :2]

        new_ego_in[:, :2] = self.convert_global_coords_to_local(coordinates=ego_in[:, :2] - translation,
                                                                yaw=angle_of_rotation)
        # new_ego_in[:, 4:6] = self.convert_global_coords_to_local(coordinates=ego_in[:, 2:4], yaw=angle_of_rotation)
        new_ego_in[:, 5:7] = self.convert_global_coords_to_local(coordinates=ego_in[:, 2:4], yaw=angle_of_rotation)
        new_ego_out[:, :2] = self.convert_global_coords_to_local(coordinates=ego_out[:, :2] - translation,
                                                                 yaw=angle_of_rotation)
        new_roads[0, :, :, :2] = self.convert_global_coords_to_local(coordinates=new_roads[0, :, :, :2] - translation,
                                                                     yaw=angle_of_rotation)
        new_roads[0][np.where(new_roads[0, :, :, -1] == 0)] = 0.0

        # other agents
        for n in range(self.num_others):
            if not agents_in[n, -1, -1]:
                continue

            if agent_types[n + 1, 0]:  # vehicle
                yaw = agents_in[n, -1, 4]
            elif agent_types[n + 1, 1]:  # pedestrian/bike
                diff = agents_in[n, -1, :2] - agents_in[n, -5, :2]
                yaw = np.arctan2(diff[1], diff[0])
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            translation = agents_in[n, -1, :2]

            new_agents_in[n, :, :2] = self.convert_global_coords_to_local(coordinates=agents_in[n, :, :2] - translation,
                                                                          yaw=angle_of_rotation)
            new_agents_in[n, :, 5:7] = self.convert_global_coords_to_local(coordinates=agents_in[n, :, 2:4],
                                                                           yaw=angle_of_rotation)
            new_agents_out[n, :, :2] = self.convert_global_coords_to_local(
                coordinates=agents_out[n, :, :2] - translation, yaw=angle_of_rotation)
            new_roads[n + 1, :, :, :2] = self.convert_global_coords_to_local(
                coordinates=new_roads[n + 1, :, :, :2] - translation, yaw=angle_of_rotation)
            new_roads[n + 1][np.where(new_roads[n + 1, :, :, -1] == 0)] = 0.0

        return new_ego_in, new_ego_out, new_agents_in, new_agents_out, new_roads

    def _plot_debug(self, ego_in, ego_out, agents_in, agents_out, roads):
        for n in range(self.num_others + 1):
            plt.figure()
            if n == 0:
                plt.scatter(ego_in[:, 0], ego_in[:, 1], color='k')
                plt.scatter(ego_out[:, 0], ego_out[:, 1], color='m')
            else:
                if agents_in[n - 1, -1, -1]:
                    plt.scatter(agents_in[n - 1, :, 0], agents_in[n - 1, :, 1], color='k')
                    plt.scatter(agents_out[n - 1, :, 0], agents_out[n - 1, :, 1], color='m')
            for s in range(roads.shape[1]):
                for p in range(roads.shape[2]):
                    if roads[n, s, p, -1]:
                        plt.scatter(roads[n, s, p, 0], roads[n, s, p, 1], color='g')
            plt.show()
        exit()

    def __getitem__(self, idx: int):
        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        agents_data = dataset['agents_trajectories'][idx]
        agent_types = dataset['agents_types'][idx]
        meta_data = dataset['metas'][idx]
        if not self.evaluation:
            agents_data = agents_data[:, 1::2]  # downsampling for efficiency during training.

        road_fname_key = dataset['map_paths'][idx][0].decode("utf-8").split("/")[-1]
        roads = self.roads[road_fname_key].copy()
        roads[:, :, :2] -= np.expand_dims(np.array([meta_data[:2]]), 0)

        original_roads = np.zeros((self.max_num_road_segs, *roads.shape[1:]))
        original_roads[:len(roads)] = roads
        roads = original_roads.copy()

        ego_in, ego_out, agents_in, agents_out, agent_types = self.split_input_output_normalize(agents_data, meta_data,
                                                                                                agent_types)
        roads = self.copy_agent_roads_across_agents(agents_in, roads)

        # normalize scenes so all agents are going up
        if self.evaluation:
            translations = np.concatenate((ego_in[-1:, :2], agents_in[:, -1, :2]), axis=0)
        ego_in, ego_out, agents_in, agents_out, roads = self.rotate_agents(ego_in, ego_out, agents_in, agents_out,
                                                                           roads, agent_types)

        '''
        Outputs:
        ego_in: One agent we are calling ego who's shape is (T x S_i) where 
        S_i = [x_loc, y_loc, yaw_loc, x_glob, y_glob, vx_loc, vy_loc, yaw_glob, length, width, existence_mask].
        agents_in: all other agents with shape of (T x num_others x S) where S is the same as above.
        ego_out: (T x S_o) where S_o = [x_loc, y_loc, x_glob, y_glob, yaw_loc, existence_mask]
        agents_out: (T x num_others x S_o)
        roads: (num_others x num_road_segs x num_pts_per_road_seg x S_r) where 
        S_r = [x_loc, y_loc, [type1,type2,type3], [left, right], existence_mask]

        '''

        if self.evaluation:
            ego_in = ego_in[1::2]
            agents_in = agents_in[:, 1::2]
            model_ego_in = ego_in.copy()
            model_ego_in[:, 3:5] = 0.0
            ego_in[:, 0:2] = ego_in[:, 3:5]
            model_agents_in = agents_in.copy()
            model_agents_in[:, :, 3:5] = 0.0
            agents_in[:, :, 0:2] = agents_in[:, :, 3:5]
            ego_out[:, 0:2] = ego_out[:, 2:4]  # putting the original coordinate systems
            agents_out[:, :, 0:2] = agents_out[:, :, 2:4]  # putting the original coordinate systems
            return model_ego_in, ego_out, model_agents_in.transpose(1, 0, 2), agents_out.transpose(1, 0, 2), roads, \
                   agent_types, ego_in, agents_in.transpose(1, 0, 2), original_roads, translations
        else:
            # Experimentally found that global information actually hurts performance.
            ego_in[:, 3:5] = 0.0
            agents_in[:, :, 3:5] = 0.0
            return ego_in, ego_out, agents_in.transpose(1, 0, 2), agents_out.transpose(1, 0, 2), roads, agent_types

    def __len__(self):
        return self.dset_len
