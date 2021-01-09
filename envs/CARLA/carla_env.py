from __future__ import print_function, division

import random
import numpy as np

from .carla_lib.carla.sensor import Camera
from .carla_lib.carla.settings import CarlaSettings
from .carla_lib.carla.image_converter import labels_to_array
from .carla_lib.carla.transform import Transform
from .carla_lib.carla.image_converter import depth_to_local_point_cloud, to_rgb_array, depth_to_logarithmic_grayscale, depth_to_array
import cv2
import math
import os
import shutil
from numpy.linalg import inv
import gc
import time
import random
import envs.CARLA.carla_utils as cutils
from envs.CARLA.carla_utils import extract_agent_bbox, tighten_bbox, vertex_3d_to_2d, default_settings, seg_to_bbox, labels_to_segimage, simplify_seg, draw_3d_bbox
from envs.CARLA.monitor_manager import MonitorManager


class CarlaEnv(object):
    def __init__(self, client, args, simple_seg=True):
        super(CarlaEnv, self).__init__()
        self.args = args
        self.client = client
        self.simple_seg = simple_seg
        self.save_record = args.save_record
        self.recorder_path = os.path.join(args.save_path, args.monitor_video_dir)
        self.episode = 0
        self.rotation = None
        self.nonplayer_ids = {}
        self.view_h = self.args.frame_height
        self.view_w = self.args.frame_width
        self.mm = MonitorManager(self.args, self.recorder_path, ["mon", "seg", "obs"], self.view_w, self.view_h)

        interval = lambda x, y: list(range(x, y+1))
        self.player_starts = interval(29, 32) + interval(34, 43) + interval(45, 54) + interval(56, 57) + interval(64, 85) + interval(87, 96) + interval(98, 107) + interval(109, 118) + interval(120, 121)

    def set_epoch(self, epoch):
        self.episode = epoch
        if epoch == 0 and os.path.isdir(self.recorder_path):
            shutil.rmtree(self.recorder_path)
            os.makedirs(self.recorder_path)

    def read_sensor(self, sensor_data, far=1.0):
        obs = np.frombuffer(sensor_data['CameraRGB'].raw_data, dtype=np.uint8).reshape((self.view_h, self.view_w, 4))[:, :, :3]
        mon = np.frombuffer(sensor_data['CameraMON'].raw_data, dtype=np.uint8).reshape((self.view_h, self.view_w, 4))[:, :, :3]
        # depth_im = depth_to_logarithmic_grayscale(sensor_data['CameraDepth'])
        # cv2.imwrite("depth_demo.png", depth_im)
        '''
        depth = np.frombuffer(sensor_data['CameraDepth'].raw_data, dtype=np.uint8).reshape((self.view_h, self.view_w, 4)).astype(np.uint32)
        R, G, B = depth[:,:,0], depth[:,:,1], depth[:,:,2]
        Dint24 = R + G*256 + B*256*256
        Ans = Dint24 / ( 256*256*256 - 1 )
        Ans = Ans * far
        '''
        depth = depth_to_array(sensor_data['CameraDepth'])
        seg = labels_to_array(sensor_data['CameraSegmentation'])
        if self.simple_seg: seg = simplify_seg(seg)
        sensor_dict = dict()
        sensor_dict['obs'] = obs
        sensor_dict['seg'] = seg
        sensor_dict['mon'] = mon
        sensor_dict['depth'] = depth
        return sensor_dict

    def signal_mapping(self, action):
        thr = action[0] # -1 ~ 1
        steer = action[1] * 0.4
        '''
        # Strategy #1: simple linear mapping from negative throttle to posistive signals
        if self.args.braking and thr < 0: 
            throttle = 0
            brake = -thr * 0.5
        else:
            throttle = thr * 0.5 + 0.5
            brake = 0
        '''
        # Strategy #2: we try to suppress the frequency of too large braking signals
        if self.args.braking and thr< 0:
            throttle = throttle
            brake = - thr
            brake = brake * brake
        else:
            throttle = thr * 0.5 + 0.5
            brake = 0

        return throttle, steer, brake

    def reset(self, testing=False):
        # set the recording related variables
        self.episode += 1  
        self.testing = testing
        self.timestep = 0
        self.collision = 0
        self.collision_vehicles = 0
        self.collision_other = 0
        self.stuck_cnt = 0
        self.collision_cnt = 0
        self.offroad_cnt = 0
        self.ignite = False
        settings, self.intrinsic, self.obs_to_car_transform = default_settings(self.args, self.view_h, self.view_w)
        self.scene = self.client.load_settings(settings)

        # spawn the player agent on randomly spawning points
        player_start = np.random.choice(self.player_starts)
        print('Starting new episode ...')
        self.client.start_episode(player_start)
        
        for _ in range(30):
            self.client.send_control(
                steer=0,
                throttle=1.0,
                brake=0.0,
                hand_brake=False,
                reverse=False)
        measurements, sensor_data = self.client.read_data()

        # read the enviornments after initial control signals
        # measurements, sensor_data = self.client.read_data()
        info = self.convert_info(measurements)
        sensor_dict = self.read_sensor(sensor_data)
        obs, info['seg'], mon = sensor_dict['obs'], sensor_dict['seg'], sensor_dict['mon']
        info['depth'] = sensor_dict['depth']
        # get bbox in the camera scene
        self.rotation = measurements.player_measurements.transform.rotation
        info = self.insert_ins_info(measurements, info)
        self.record(obs, labels_to_segimage(info["seg"]), mon, info)
        
        return obs, info


    def step(self, action=None, expert=False, rnd=0):
        self.timestep += 1

        if expert:
            self.client.send_control(action)
        else:
            throttle, steer, brake = self.signal_mapping(action)
            if abs(steer) < self.args.steer_clip:
                steer = 0
            self.client.send_control(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False)
        measurements, sensor_data = self.client.read_data()

        info = self.convert_info(measurements)
        sensor_dict = self.read_sensor(sensor_data)
        obs, info['seg'], mon, info['depth'] = sensor_dict['obs'], sensor_dict['seg'], sensor_dict['mon'], sensor_dict['depth']

        self.rotation = measurements.player_measurements.transform.rotation
        
        info = self.insert_ins_info(measurements, info)
        done = self.done_from_info(info) or self.timestep > 1000
        reward = self.reward_from_info(info)
        self.record(obs, labels_to_segimage(info["seg"]), mon, info)

        if done and self.save_record:
            for subdir in ["mon", "seg", "obs"]: 
                self.mm.merge(subdir, self.episode, self.timestep+1)
      
        return obs, reward, done, info

    def get_bbox(self, measurement, seg, coll_veh_num):
        width = self.view_w
        height = self.view_h
        player_transform = measurement.player_measurements.transform
        extrinsic = Transform(player_transform) * self.obs_to_car_transform 
        bbox_list = []
        rotation_list = []
        distance_list = []
        dimensions_list = []
        player_location = measurement.player_measurements.transform.location
        player_location = np.array([player_location.x, player_location.y, player_location.z])
        bboxes_3d = []
        eight_vertices= []
        # collect the 2D-bbox generated from the 3d-bbox of non-player agents
        for agent in measurement.non_player_agents:
            if agent.HasField("vehicle"):
                bbox, rotation, vehicle_location, dimensions = extract_agent_bbox(agent)
                cur_location = np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z])
                distance = np.linalg.norm(player_location - cur_location)
                    
                vertices = []
                tmp_vertices = []
                for vertex in bbox:
                    pos2d = vertex_3d_to_2d(vertex, self.intrinsic, extrinsic) # [X, Y, Z]    
                    x_2d = width - pos2d[0]
                    y_2d = height - pos2d[1]
                    tmp_vertices.append([x_2d, y_2d, pos2d[2]])
                    if pos2d[2] > 0:
                        # on the observation plane
                        vertices.append([x_2d, y_2d, pos2d[2]])


                if len(vertices) > 1:
                    vertices = np.array(vertices)
                    padded_x1 = np.min(vertices[:, 0])
                    padded_y1 = np.min(vertices[:, 1])
                    padded_x2 = np.max(vertices[:, 0])
                    padded_y2 = np.max(vertices[:, 1])
                    # append both the 2d and 3d bounding box annotations
                    eight_vertices.append(tmp_vertices)
                    bbox_list.append([padded_x1, padded_y1, padded_x2, padded_y2])
                    rotation_list.append(rotation.yaw)
                    distance_list.append(distance)
                    dimensions_list.append(dimensions)
        seg_bboxes = seg_to_bbox(seg)

        tight_bboxes, bboxes_indices, visible_bboxes3d = tighten_bbox(bbox_list, eight_vertices, seg_bboxes, width, height)
        # only selected agents' instance-level information will be returned: bboxes_indices
        rotations = np.array(rotation_list)[bboxes_indices]
        distances = np.array(distance_list)[bboxes_indices]
        dimensionses = np.array(dimensions_list)[bboxes_indices]

        '''
        as CARLA 0.8.4 doesn't provide the API to get enough non-player information, so we can't
        directly know which vehicles the ego-vehicle collides with, as a comprimising solution,
        when collision with k vehicles is detected on one frame, we sign the collision to the k nearest
        vehicles with the ego-vehicle.
        '''

        visible_coll_num = min(coll_veh_num, distances.size)
        coll_idx = np.argpartition(distances, visible_coll_num-1)[:visible_coll_num]
        coll_with = np.zeros(distances.size)
        coll_with[coll_idx] = 1

        calib = dict()
        calib["intrinsic"] = self.intrinsic
        calib["extrinsic"] = np.array(extrinsic.matrix)
        calib["player_transform"] = np.array(Transform(player_transform).matrix)
        calib["camera_transform"] = np.array(self.obs_to_car_transform.matrix)

        return tight_bboxes, distances, coll_with, visible_bboxes3d, rotations, dimensionses, calib


    def insert_ins_info(self, measurements, info):
        if self.args.use_detection:
            info["bboxes"], info["distances"], info['coll_with'], info["3d_bboxes"], info["rotations"], info["dimensionses"], info["calib"] = \
            self.get_bbox(measurements, info["seg"], info['coll_veh_num'])
            info['orientations'] = info['rotations'] - self.rotation.yaw
        else:
            info["bboxes"] = None 
            info["distances"] = None
            info['coll_with'] = None
            info["3d_bboxes"] = None 
            info["rotations"] = None 
            info["dimensionses"] = None 
            info["calib"] = None 
            info['orientations'] = None
        
        return info

    def record(self, obs, seg, mon, info=dict()):
        width = self.view_w
        height = self.view_h
        obs = obs.astype(np.int32)
        if self.save_record: 
            self.mm.save("mon", self.episode, self.timestep, mon, info)
            self.mm.draw_and_save("obs", self.episode, self.timestep, obs, info)
            self.mm.save("seg", self.episode, self.timestep, seg)

    def convert_info(self, measurements):
        info = dict()
        info['speed'] = measurements.player_measurements.forward_speed
        info['collision'] = int(measurements.player_measurements.collision_other + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_vehicles > self.collision or (self.collision_cnt > 0 and info['speed'] < 0.5))
        info['collision_other'] = int(measurements.player_measurements.collision_other > self.collision_other)
        info['collision_vehicles'] = int(measurements.player_measurements.collision_vehicles > self.collision_vehicles)
        info['coll_veh_num'] = int(measurements.player_measurements.collision_vehicles - self.collision_vehicles)
        self.collision_vehicles = measurements.player_measurements.collision_vehicles
        self.collision_other = measurements.player_measurements.collision_other 
        self.collision = measurements.player_measurements.collision_other + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_vehicles
        info['offlane'] = int(measurements.player_measurements.intersection_otherlane > 0.01)
        info['offroad'] = int(measurements.player_measurements.intersection_offroad > 0.001)
        info['expert_control'] = measurements.player_measurements.autopilot_control
        return info

    def done_from_info(self, info):
        if info['collision'] > 0 or (self.collision_cnt > 0 and info['speed'] < 0.5):
            self.collision_cnt += 1
        else:
            self.collision_cnt = 0
        self.ignite = self.ignite or info['speed'] > 1
        stuck = int(info['speed'] < 1)
        self.stuck_cnt = (self.stuck_cnt + stuck) * stuck * int(bool(self.ignite) or self.testing)

        if info['offroad'] > 0.5:
            self.offroad_cnt += 1

        return (self.stuck_cnt > 30) or self.offroad_cnt > 30 or self.collision_cnt > 20

    def reward_from_info(self, info):
        reward = dict()
        reward['without_pos'] = info['speed'] / 15 - info['offroad'] - info['collision'] * 2.0
        reward['with_pos'] = reward['without_pos'] - info['offlane'] / 5
        return reward['with_pos']

        
