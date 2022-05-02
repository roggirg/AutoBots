import cv2
import os
import math

from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from typing import Dict


from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.input_representation.static_layers import get_lanes_in_radius, StaticLayerRasterizer, correct_yaw
from nuscenes.prediction.helper import quaternion_yaw, convert_global_coords_to_local

import matplotlib.pyplot as plt
import numpy as np


def is_insquare(points, car_pos, ego_range):
    if (car_pos[0] - ego_range[0] < points[0] < car_pos[0] + ego_range[1]) and \
        (car_pos[1] - ego_range[2] < points[1] < car_pos[1] + ego_range[3]):
        return True
    return False


def distance(t1, t2) -> float:

    return math.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)


def load_all_maps(helper: PredictHelper, verbose: bool = False) -> Dict[str, NuScenesMap]:
    """
    Loads all NuScenesMap instances for all available maps.
    :param helper: Instance of PredictHelper.
    :param verbose: Whether to print to stdout.
    :return: Mapping from map-name to the NuScenesMap api instance.
    """
    dataroot = helper.data.dataroot

    json_files = filter(
        lambda f: "json" in f and "prediction_scenes" not in f, os.listdir(os.path.join(dataroot, "maps", "expansion"))
    )
    maps = {}

    for map_file in json_files:

        map_name = str(map_file.split(".")[0])
        if verbose:
            print(f"static_layers.py - Loading Map: {map_name}")

        maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

    return maps


class NuScenesDataset(Dataset):
    def __init__(self, data_root, split_name, version, ego_range=(25, 25, 10, 50), debug=False, num_others=10):

        super(NuScenesDataset).__init__()
        nusc = NuScenes(version=version, dataroot=data_root)
        self._helper = PredictHelper(nusc)
        self._dataset = get_prediction_challenge_split(split_name, dataroot=data_root)
        self._use_pedestrians = False
        self._use_only_moving_vehicles = True
        self._debug = debug
        self.ego_range = ego_range  # (left, right, behind, front)
        # parameters
        self.future_secs = 6
        self.past_secs = 3
        self.freq_hz = 2
        self._number_closest_agents = num_others  # We always take this number of closest objects
        self._max_number_roads = 100
        self._number_future_road_points = 40  # the number of points to include along the current lane.
        self._map_dict = load_all_maps(self._helper, verbose=True)
        self._static_layer_rasterizer = StaticLayerRasterizer(
            self._helper,
            layer_names=['drivable_area', 'ped_crossing', 'walkway'],
            resolution=0.2,
            meters_ahead=ego_range[3],
            meters_behind=ego_range[2],
            meters_left=ego_range[0],
            meters_right=ego_range[1],
        )

        if self._debug:
            self._static_layer_rasterizer_1 = StaticLayerRasterizer(
                self._helper,
                resolution=0.1,
                meters_ahead=100,
                meters_behind=50,
                meters_left=75,
                meters_right=75,
            )
            self._agent_rasterizer = AgentBoxesWithFadedHistory(
                self._helper,
                seconds_of_history=1,
                resolution=0.1,
                meters_ahead=100,
                meters_behind=50,
                meters_left=75,
                meters_right=75,
            )
            self._mtp_input_representation = InputRepresentation(
                self._static_layer_rasterizer_1, self._agent_rasterizer, Rasterizer()
            )

    def debug_draw_full_map_from_position(self, data_root, instance, sample):

        image_folder = os.path.join(data_root, instance)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)

        future = self._helper.get_future_for_agent(instance, sample, seconds=200, in_agent_frame=True, just_xy=False)
        past = self._helper.get_past_for_agent(instance, sample, seconds=200, in_agent_frame=True, just_xy=False)

        count = 0

        for token in reversed(past):
            img = self._mtp_input_representation.make_input_representation(instance, token["sample_token"])
            plt.imshow(img)
            plt.savefig(os.path.join(image_folder, str(count) + ".png"))
            count += 1

        for token in future:
            img = self._mtp_input_representation.make_input_representation(instance, token["sample_token"])
            plt.imshow(img)
            plt.savefig(os.path.join(image_folder, str(count) + ".png"))
            count += 1

    def _get_map_features(self, nusc_map, x, y, yaw, radius, reference_position):
        curr_map = np.zeros((self._max_number_roads, self._number_future_road_points, 4))
        lanes = get_lanes_in_radius(x=x, y=y, radius=200, map_api=nusc_map, discretization_meters=2.0)

        # need to combine lanes that are connected to avoid random gaps.
        combined_lane_ids = []  # list of connected lane ids.
        ignore_lane_ids = []  # List of lane ids to ignore because they are connected to prior lane ids.
        for lane_id in lanes.keys():
            if lane_id in ignore_lane_ids:
                continue
            curr_lane_ids = [lane_id]
            out_lane_ids = nusc_map.get_outgoing_lane_ids(lane_id)
            for out_lane_id in out_lane_ids:
                if out_lane_id in lanes.keys():
                    curr_lane_ids.append(out_lane_id)
                    ignore_lane_ids.append(out_lane_id)

                    outout_lane_ids = nusc_map.get_outgoing_lane_ids(out_lane_id)
                    for outout_lane_id in outout_lane_ids:
                        if outout_lane_id in lanes.keys():
                            curr_lane_ids.append(outout_lane_id)
                            ignore_lane_ids.append(outout_lane_id)

            combined_lane_ids.append(curr_lane_ids)

        relevant_pts = []
        for i in range(len(combined_lane_ids)):
            for lane_id in combined_lane_ids[i]:
                pts = lanes[lane_id]
                road_pts = []
                for pt in pts:
                    pt = list(pt)
                    pt.append(1.0)
                    pt = np.array(pt)
                    road_pts.append(pt)
                if len(road_pts) > 0:
                    relevant_pts.append(np.array(road_pts))

        # relevant_pts is a list of lists, which each sublist containing the points on a road in the fov.
        if len(relevant_pts) > 0:
            # relevant_pts is a list of arrays each with [numb_pts, 3]
            # need to sort each one according to its closeness to the agent position
            first_pts = []
            for i in range(len(relevant_pts)):
                indices = np.argsort(np.linalg.norm(np.array([[x, y]]) - relevant_pts[i][:, :2], axis=1))
                relevant_pts[i] = relevant_pts[i][indices]
                first_pts.append(relevant_pts[i][0])

            # sort using the first point of each road
            first_pts = np.array(first_pts)
            indices = np.argsort(np.linalg.norm(np.array([[x, y]]) - first_pts[:, :2], axis=1)).tolist()
            mymap = np.array(relevant_pts, dtype=object)[indices[:self._max_number_roads]]
            for i in range(min(len(mymap), self._max_number_roads)):
                curr_map[i, :min(len(mymap[i]), self._number_future_road_points)] = \
                    mymap[i][:min(len(mymap[i]), self._number_future_road_points)]
        else:
            raise Exception("Did not find any lanes in the map...")

        return curr_map

    def _rotate_sample_points(self, annotation, sample_info):
        # Rotate all points using ego translation and rotation
        del sample_info[annotation['instance_token']]  # Remove ego data from here
        for key, val in list(sample_info.items()):
            if val.size == 0:
                # delete empty entries.
                del sample_info[key]
                continue
            sample_info[key] = convert_global_coords_to_local(val, annotation['translation'], annotation['rotation'])
        return sample_info

    def choose_agents(self, past_samples, future_samples, sample_token):
        # Rectangle around ego at the prediction time
        x_left, x_right = -self.ego_range[0], self.ego_range[1]  # in meters
        y_behind, y_infront = -self.ego_range[2], self.ego_range[3]  # in meters
        # x_left, x_right = -25, 25  # in meters
        # y_behind, y_infront = -10, 30  # in meters

        agent_types = {}
        # we only want to consider at the prediction timetep, last point from the past(first in array).
        valid_agent_ids = {}
        for key, value in past_samples.items():
            info = self._helper.get_sample_annotation(key, sample_token)
            agent_type = info['category_name']
            useful_agent_bool = "vehicle" in agent_type or "human" in agent_type
            if x_left <= value[0, 0] <= x_right and y_behind <= value[0, 1] <= y_infront and \
                    key in future_samples.keys() and useful_agent_bool:
                dist_to_ego_at_t = distance(value[0], [0,0])
                valid_agent_ids[key] = dist_to_ego_at_t
                agent_types[key] = agent_type

        # sort according to distance to ego at t.
        valid_agent_ids = [k for k, v in sorted(valid_agent_ids.items(), key=lambda item: item[1])]
        if len(valid_agent_ids) > self._number_closest_agents:
            final_valid_keys = valid_agent_ids[:self._number_closest_agents]
        else:
            final_valid_keys = valid_agent_ids

        # construct numpy array for all agent points
        agents_types_list = []
        agents_array = np.zeros((self._number_closest_agents, (self.past_secs+self.future_secs)*2, 3))
        for i, key in enumerate(final_valid_keys):
            # flipped past samples like before
            curr_agent = np.concatenate((past_samples[key][::-1], future_samples[key]), axis=0)
            if len(curr_agent) > 18:
                print("too many timesteps...")
                curr_agent = curr_agent[:(self.future_secs + self.past_secs) * self.freq_hz]
            agents_array[i, :len(curr_agent), :2] = curr_agent
            agents_array[i, :len(curr_agent), 2] = 1
            agents_types_list.append(agent_types[key])

        return agents_array, agents_types_list

    def __getitem__(self, idx: int):
        instance_token, sample_token = self._dataset[idx].split("_")
        annotation = self._helper.get_sample_annotation(instance_token, sample_token)
        ego_type = [annotation['category_name']]
        road_img = self._static_layer_rasterizer.make_representation(instance_token, sample_token)
        road_img = cv2.resize(road_img, (750, 750))

        # Ego-Agent stuff
        p_all_positions = self._helper.get_past_for_agent(
            instance_token, sample_token, seconds=self.past_secs, in_agent_frame=False, just_xy=True
        )
        f_all_positions = self._helper.get_future_for_agent(
            instance_token, sample_token, seconds=self.future_secs, in_agent_frame=False, just_xy=True
        )

        if self._debug:
            self.debug_draw_full_map_from_position(
                data_root=".", instance=instance_token, sample=sample_token
            )

        all_ego_positions = np.concatenate((p_all_positions[::-1], f_all_positions), axis=0)  # need to flip past.
        rotated_ego_positions = []
        for coords in all_ego_positions:
            rotated_ego_positions.append(
                convert_global_coords_to_local(coords, annotation['translation'], annotation['rotation']).squeeze())
        rotated_ego_positions = np.array(rotated_ego_positions)
        ego_array = np.zeros(((self.future_secs+self.past_secs)*self.freq_hz, 3))  # 3 for x,y,mask
        if len(rotated_ego_positions) > 18:
            print("too many timesteps...")
            rotated_ego_positions = rotated_ego_positions[:(self.future_secs+self.past_secs)*self.freq_hz]
        ego_array[:len(rotated_ego_positions), :2] = rotated_ego_positions
        ego_array[:len(rotated_ego_positions), 2] = 1

        # Other agents stuff
        p_sample_info = self._helper.get_past_for_sample(
            sample_token, seconds=self.past_secs, in_agent_frame=False, just_xy=True
        )
        f_sample_info = self._helper.get_future_for_sample(
            sample_token, seconds=self.future_secs, in_agent_frame=False, just_xy=True
        )
        p_sample_info = self._rotate_sample_points(annotation, p_sample_info)
        f_sample_info = self._rotate_sample_points(annotation, f_sample_info)
        agents_array, agent_types = self.choose_agents(p_sample_info, f_sample_info, sample_token)
        agent_types = ego_type + agent_types

        # Map stuff
        map_name = self._helper.get_map_name_from_sample_token(sample_token)
        theta = correct_yaw(quaternion_yaw(Quaternion(annotation['rotation'])))
        raw_map = self._get_map_features(self._map_dict[map_name], annotation["translation"][0], annotation["translation"][1], theta, 100, [0, 0])

        rotated_map = np.zeros_like(raw_map)
        for road_idx in range(len(raw_map)):
            for pt_idx in range(len(raw_map[road_idx])):
                if raw_map[road_idx, pt_idx, -1] == 1:
                    rotated_pt = convert_global_coords_to_local(raw_map[road_idx, pt_idx, :2], annotation['translation'],
                                                                annotation['rotation'])
                    if is_insquare(rotated_pt[0], [0., 0.], self.ego_range):
                        rotated_map[road_idx, pt_idx, :2] = rotated_pt
                        rotated_map[road_idx, pt_idx, 2] = theta - raw_map[road_idx, pt_idx, 2]
                        rotated_map[road_idx, pt_idx, -1] = 1

        extras = [instance_token, sample_token, annotation['translation'], annotation['rotation'], map_name]
        return ego_array, agents_array.transpose((1, 0, 2)), road_img, extras, agent_types, rotated_map

    def __len__(self):
        return len(self._dataset)

