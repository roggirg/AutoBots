#!/usr/bin/env python

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

import xml.etree.ElementTree as xml
import pyproj
import math

import sys
import numpy as np
import cv2


def get_value_list(d):
    assert isinstance(d, dict)
    if sys.version_info[0] == 2:
        item_list = d.values()
    elif sys.version_info[0] == 3:
        item_list = list(d.values())
    else:
        # should not happen
        raise RuntimeError("Only python 2 and 3 supported.")
    assert isinstance(item_list, list)
    return item_list


def get_item_iterator(d):
    assert isinstance(d, dict)
    if sys.version_info[0] == 2:
        item_iter = d.iteritems()
        assert hasattr(item_iter, "next")
    elif sys.version_info[0] == 3:
        item_iter = iter(d.items())
        assert hasattr(item_iter, "__next__")
    else:
        # should not happen
        raise RuntimeError("Only python 2 and 3 supported.")
    assert hasattr(item_iter, "__iter__")
    return item_iter


class Point:
    def __init__(self):
        self.x = None
        self.y = None


class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]


def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None


def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None


def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list


def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])


def get_minmax(point_dict):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9
    for id, point in get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)
    return min_x, min_y, max_x, max_y


def get_relation_members(rel, lane_dict, exclusion_ids):
    found_lanelet = False
    for tag in rel.findall("tag"):
        if tag.get("v") == "lanelet":
            found_lanelet = True

    if not found_lanelet:
        return None, None

    lanes = []
    used_lane_ids = []
    for member in rel.findall("member"):
        if member.get("type") == "way":
            lane_id = member.get("ref")
            if lane_id in exclusion_ids:
                return None, None
            used_lane_ids.append(lane_id)

            lane_role = member.get("role")
            lane_role_np = np.zeros((40, 2))
            if "left" in lane_role:
                lane_role_np[:, 0] += 3.0
            elif "right" in lane_role:
                lane_role_np[:, 1] += 3.0

            curr_lane = np.zeros((40, 8))
            curr_lane[:, :5] = lane_dict[lane_id][:, :5]  # state (position (2) and type (3))
            curr_lane[:, 5:7] = lane_role_np  # relationship (left, right)
            curr_lane[:, -1] = lane_dict[lane_id][:, -1]  # existence mask
            lanes.append(curr_lane)

    return lanes, used_lane_ids


def get_minmax_mapfile(filename):
    projector = LL2XYProjector(0.0, 0.0)

    e = xml.parse(filename).getroot()

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    xmin, ymin, xmax, ymax = get_minmax(point_dict)
    return xmin, ymin, xmax, ymax
