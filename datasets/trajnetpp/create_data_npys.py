import argparse
import os
import numpy as np
import trajnetplusplustools
import math


'''
This is only for the synthetic portion of the TrajNet++ dataset. It can be modified to also create files for the real 
portion.
'''


def drop_distant(xy, max_num_peds=5):
    """
    Only Keep the max_num_peds closest pedestrians
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    smallest_dist_to_ego = np.nanmin(distance_2, axis=0)
    return xy[:, np.argsort(smallest_dist_to_ego)[:(max_num_peds)]]


def drop_inactive(xy, obs_horizon=9):
    """
    Only keep agents that are active at the last timestep in the past.
    """
    return xy[:, ~np.isnan(xy[obs_horizon-1, :, 0])]


def shift(xy, center):
    # theta = random.random() * 2.0 * math.pi
    xy = xy - center[np.newaxis, np.newaxis, :]
    return xy


def theta_rotation(xy, theta):
    # theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = np.array([[ct, st], [-st, ct]])
    return np.einsum('ptc,ci->pti', xy, r)


def center_scene(xy, obs_length=9, ped_id=0):
    ## Center
    center = xy[obs_length - 1, ped_id]  ## Last Observation
    xy = shift(xy, center)

    ## Rotate
    last_obs = xy[obs_length - 1, ped_id]
    second_last_obs = xy[obs_length - 2, ped_id]
    diff = np.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
    thet = np.arctan2(diff[1], diff[0])
    rotation = -thet + np.pi / 2
    xy = theta_rotation(xy, rotation)
    return xy, rotation, center


def inverse_scene(xy, rotation, center):
    xy = theta_rotation(xy, -rotation)
    xy = shift(xy, -center)
    return xy


def get_args():
    parser = argparse.ArgumentParser(description="TrajNet++ NPY Creator")
    parser.add_argument("--output-npy-path", type=str, required=True, help="output path to H5 files.")
    parser.add_argument("--raw-dataset-path", type=str, required=True, help="raw Dataset path to .../synth_data.")
    args = parser.parse_args()
    return args


def prepare_data(raw_path, out_path):
    N = 6
    files = [f.split('.')[-2] for f in os.listdir(raw_path) if f.endswith('.ndjson')]
    for file in files:
        reader = trajnetplusplustools.Reader(raw_path + file + '.ndjson', scene_type='tags')
        scene = [(file, s_id, s_tag, xypaths) for s_id, s_tag, xypaths in reader.scenes(sample=1.0)]

        val_len = int(len(scene)*0.1)
        train_len = len(scene) - val_len
        val_file_data = np.zeros((val_len, 21, N, 2))
        train_file_data = np.zeros((train_len, 21, N, 2))
        largest_num_agents = 0.0
        for scene_i, (filename, scene_id, s_tag, curr_scene) in enumerate(scene):
            curr_scene = drop_distant(curr_scene, max_num_peds=N)
            curr_scene, _, _ = center_scene(curr_scene)

            if curr_scene.shape[1] > largest_num_agents:
                largest_num_agents = curr_scene.shape[1]

            if curr_scene.shape[1] < N:
                # Need to pad array to have shape 21xNx2
                temp_curr_scene = np.zeros((21, N, 2))
                temp_curr_scene[:, :, :] = np.nan
                temp_curr_scene[:, :curr_scene.shape[1], :] = curr_scene
                curr_scene = temp_curr_scene.copy()

            if scene_i < val_len:
                val_file_data[scene_i] = curr_scene
            else:
                train_file_data[scene_i - val_len] = curr_scene

        np.save(os.path.join(out_path, "val_"+file+".npy"), val_file_data)
        np.save(os.path.join(out_path, "train_"+file+".npy"), train_file_data)
        del val_file_data
        del train_file_data
        del scene
        del reader
        print("FILE", file, "Largest_num_agents", largest_num_agents)


if __name__ == "__main__":
    args = get_args()
    prepare_data(raw_path=args.raw_dataset_path, out_path=args.output_npy_path)
