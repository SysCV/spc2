# this is a script to use built-in autopilot agent for data collection
# nothing to do with what we proposed in the project
import argparse
from args import init_parser, post_processing
import numpy as np
from envs import make_env

# find the carla module 
import glob
import os
import sys
import cv2
import math 
import random
import time
import torch
from envs.CARLA.carla_lib.carla.client import make_carla_client
from envs.CARLA.carla_env import CarlaEnv


parser = argparse.ArgumentParser(description='SPC')
init_parser(parser)  # See `args.py` for default arguments
args = parser.parse_args()
args = post_processing(args)
args.recording_frame = True
args.monitor = True
args.use_depth = True
args.port = 2666
args.vehicle_num = 128
args.use_detection = True
args.frame_width = 512
args.frame_height = 256
args.use_3d_detection = True

save_path = 'carla_datasetv4'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

if not os.path.isdir(args.save_path):
    os.makedirs(os.path.join(args.save_path, "monitor_record", "obs"))
    os.makedirs(os.path.join(args.save_path, "monitor_record", "mon"))
    os.makedirs(os.path.join(args.save_path, "monitor_record", "seg"))
else:
    if args.debug:
        import shutil
        shutil.rmtree(args.save_path)


def save(obs, info, control, episode, step):
    episode_path = os.path.join(save_path, str(episode))
    f_action = open(os.path.join(episode_path, 'action.txt'), 'a')
    f_states = open(os.path.join(episode_path, 'state.txt'), 'a')
    f_collwiths = open(os.path.join(episode_path, 'coll_withs.txt'), 'a')
    img_path = os.path.join(episode_path, "obs", '{}.jpg'.format(step))
    cv2.imwrite(img_path, obs)
    np.save(os.path.join(episode_path, "seg", '{}.npy'.format(step)), info['seg'])

    # save visible 3d bounding boxes
    if args.use_3d_detection:
        bbox_3d = info["3d_bboxes"]
        bbox_3d = np.array(bbox_3d).reshape(-1,8,3) if len(bbox_3d) > 0 else np.zeros(1)
        np.save(os.path.join(episode_path, "3d_bbox", "{}.npy".format(step)), bbox_3d)

    if args.use_detection:
        bboxes = info["bboxes"]
        bboxes = np.array(bboxes) if len(bboxes) > 0 else np.zeros(1)
        np.save(os.path.join(episode_path, "2d_bbox", "{}.npy".format(step)), bboxes)

    if args.use_depth:
        depth = info["depth"]
        np.save(os.path.join(episode_path, "depth", "{}.npy".format(step)), depth)

    orientations = info['orientations']
    dimensionses = info['dimensionses']
    calib = info["calib"]
    np.save(os.path.join(episode_path, "calib", "intrinsic_{}.npy".format(step)), calib["intrinsic"])
    np.save(os.path.join(episode_path, "calib", "extrinsic_{}.npy".format(step)), calib["extrinsic"])
    np.save(os.path.join(episode_path, "calib", "player_transform_{}.npy".format(step)), calib["player_transform"])
    np.save(os.path.join(episode_path, "calib", "camera_transform_{}.npy".format(step)), calib["camera_transform"])
    # player_x, player_y, player_z = extrinsic.location.x, extrinsic.location.y, extrinsic.location.z
    # player_orix, player_oriy = extrinsic.orientation.x, extrinsic.orientation.y
    # player_yaw = extrinsic.rotation.yaw
    # extrinsic = np.array([player_x, player_y, player_z, player_orix, player_oriy, player_yaw])
    colls_with = info['coll_with']
    np.save(os.path.join(episode_path, "orientations", "{}.npy".format(step)), orientations)
    np.save(os.path.join(episode_path, "dimensions", "{}.npy".format(step)), dimensionses)
    # np.save(os.path.join(episode_path, "extrinsics", "{}.npy".format(step)), extrinsic)

    f_collwiths.write("{}: {}\n".format(step, colls_with))
    f_action.write('{}: {} {}\n'.format(step, control.steer, control.throttle))
    f_states.write('{}: {} {} {} {}\n'.format(step, info['collision'], info['collision_other'], info['offroad'], info['offlane']))

def loop(client, step, episode):
    obs, info, control = client.reset(autopilot=True)
    # save(obs, info, control, episode, 0)
    for i in range(step):
        obs, info, done, control, _ = client.step(autopilot=True, rnd=0.07)
        episode_path = os.path.join(save_path, str(episode))
        if i == 0:
            os.makedirs(os.path.join(episode_path, "obs"))
            os.makedirs(os.path.join(episode_path, "seg"))
            os.makedirs(os.path.join(episode_path, "2d_bbox"))
            os.makedirs(os.path.join(episode_path, "3d_bbox"))
            os.makedirs(os.path.join(episode_path, "depth"))
            os.makedirs(os.path.join(episode_path, "orientations"))
            os.makedirs(os.path.join(episode_path, "dimensions"))
            os.makedirs(os.path.join(episode_path, "calib"))
        save(obs, info, control, episode, i + 1)
        if done:
            print("finished at step {}".format(i))
            return


def main():
    client = make_carla_client('localhost', args.port, 100000)
    env = CarlaEnv(client, args)
    for i in range(200):
        print("===== begin episode {} =====".format(i))
        loop(env, 2000, i)

if __name__ == '__main__':
    main()
