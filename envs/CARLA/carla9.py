# this version supports CARLA 0.9.x (x<6) well but runs with some issues with x>6
# for CARLA 0.9.x (x>5), the python api lib could not be installed by pip
# but only supports manually compiling from source code
import numpy as np
import re
import random
import math
import copy
import os
import time
from envs.CARLA.sensor import CollisionSensor, LaneInvasionSensor, CameraManager
import shutil
import sys
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib
from utils import monitor_guide
# import pygame
# from .bbox import ClientSideBoundingBoxes, crop_visible_bboxes
from .carla_env import MonitorManager
from .carla_lib.carla.image_converter import labels_to_array

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# NOTE !: import carla only works on CARLA9 before 0.9.6,
# as the later version does not support lib install by pip
# users are expected to build the PythonAPI lib by manually 
# compiling from the source code
import carla
from agents.navigation.roaming_agent import *
from agents.navigation.basic_agent import *


RGB_CAMERA_INDEX = 0
SEG_CAMERA_INDEX = 5

class Buffer():
    def __init__(self, sizes):
        # sizes is a list of size by order, for example [B, C, H, W]
        self.buffer = np.zeros(sizes)
        self.buffer_len = sizes[0]
        self.full = False 
        self.cur = 0

    def insert(self, data):
        self.buffer[cur] = data
        if self.cur == self.buffer_len - 1:
            self.cur = 0
            self.full = True
        else:
            self.cur += 1
        
    def get(self, index=0):
        # get the index-th last data from buffer
        if self.cur >= index:
            return self.buffer[self.cur-index]
        else:
            if self.full and (self.buffer_len) > index:
                return self.buffer[self.buffer_len+self.cur-index]
            elif not self.full:
                return self.buffer[0]
            else:
                assert(0) # the buffer is too short
    
    
class data_buffer():
    # this class is defined to help sync data from GPU and CPU in CARLA
    def __init__(self, cpu_sensors, gpu_sensors, frame_gap=2, max_len=100):
        self.cpu_sensor_num = len(cpu_sensors)
        self.gpu_sensor_num = len(gpu_sensors)
        self.frame_gap = frame_gap
        self.buffer_len = max_len
        self.create_buffers(cpu_sensors, gpu_sensors)

    def create_rgb_buffer(self, rgb_sensor):
        image_x = rgb_sensor.image_size_x
        image_y = rgb_sensor.image_size_y
        buffer = Buffer([self.buffer_len, 3, image_x, image_y])
        return buffer

    def create_seg_buffer(self, seg_sensor):
        # by default, carla supports 13 categories of segmentation labels
        # altough we would simplify it by merging some categories, we dont do it here
        image_x = rgb_sensor.image_size_x
        image_y = rgb_sensor.image_size_y
        buffer = Buffer([self.buffer_len, image_x, image_y, 13])
        return buffer

    def create_buffers(self, cpu_sensors, gpu_sensors):
        self.buffers = dict()
        for sensor in gpu_sensors:
            if sensor.blueprint == sensor.camera.rgb:
                buffer = self.create_rgb_buffer(sensor)
                self.buffers[sensor.id] = buffer
            elif sensor.blueprint == sensor.camera.semantic_segmentation:
                buffer = self.create_seg_buffer(sensor)
                self.buffers[sensor.id] = buffer
            else:
                assert(0) # for now, we dont use other gpu sensors
        
        for sensor in cpu_sensors:
            # Lane invasion detector or Collision detector
            buffer = Buffer([self.buffer_len, 1])
            self.buffers[sensor.id] = buffer

    def insert(self, sensor, data):
        assert(sensor.id in self.buffers.keys())
        self.buffers[sensor.id].insert(data)

    def get(self, sensor, index=0):
        assert(sensor.id in self.buffers.keys())
        return self.buffers[sensor.id].get(index)


def customize_weather(cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0, 
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0, wetness=0.0, 
            sun_azimuth_angle=0.0, sun_altitude_angle=0.0):
    # this function allows to customize weather by several parameters
    '''
    cloudiness (float) – 0 is a clear sky, 100 complete overcast.
    precipitation (float) – 0 is no rain at all, 100 a heavy rain.
    precipitation_deposits (float) – 0 means no puddles on the road, 100 means roads completely capped by rain.
    wind_intensity (float) – 0 is calm, 100 a strong wind.
    sun_azimuth_angle (float) – 0 is an arbitrary North, 180 its corresponding South.
    sun_altitude_angle (float) – 90 is midday, -90 is midnight.
    fog_density (float) – Density of the fog, from 0 to 100.
    fog_distance (float) – Distance where the fog starts in meters.
    wetness (float) – Humidity percentages of the road, from 0 to 100.
    '''
    weather = carla.WeatherParameters(cloudiness=cloudiness, precipitation=precipitation,
            precipitation_deposits=precipitation_deposits, wind_intensity=wind_intensity,
            fog_density=fog_density, fog_distance=fog_distance, wetness=wetness,
            sun_azimuth_angle=sun_azimuth_angle, sun_altitude_angle=sun_altitude_angle)
    return weather

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class World(object):
    def __init__(self, args, carla_world):
        print("Begin to iniltialize the world")
        self.args = args
        self.world = carla_world
        self.map = self.world.get_map()
        self.player = None
        self.player2 = None
        self.word_port = self.args.port
        # self.clock = pygame.time.Clock()
        self.ts = 0
        self.frame = 0
        print("11")
        self._weather_presets = find_weather_presets()
        self._weather_index = self.args.weather_id
        self.vehicle_num = self.args.vehicle_num
        self.ped_num = self.args.ped_num

        self.view_h = self.args.frame_height
        self.view_w = self.args.frame_width

        # queues to store camera snapshots
        # self.obs_q = []
        # self.seg_q = []
        # self.mon_q = []
        self.guide_q = []
        self.guide_dis_q = []
        self.timestamp_q = []
        self.bbox_q = []
        self.bbox_3d_buffer = []
        self.rotations_buffer = []
        self.save_record = self.args.save_record
        self.recorder_path = os.path.join(self.args.save_path, self.args.monitor_video_dir)
        self.mm = MonitorManager(self.recorder_path, ['mon', 
        'seg', 'obs'], self.view_w, self.view_h)

        # sensors and camera
        self.rgb_camera = None
        self.seg_camera = None
        self.col_sensor = None
        self.offroad_detector = None

        # global environments
        self.vehicles = []
        self.peds = []
        self.timestamp = 0
        self.episode = 0
        self.log = open(os.path.join(self.args.save_path, 'action_info_log.txt'), 'w')

        # variables for training policy
        self.collision = False
        self.offroad = False
        self.offlane = False
        self.stuck_cnt = 0
        self.collision_cnt = 0
        self.offroad_cnt = 0
        self.ignite = False
        self.reward = 0.0
        self.crosssolid_cnt = 0
        # self.clock = clock = pygame.time.Clock()
    
        # start the first episode
        self.npc_list = []
        self.npc_waypoint_list = []
        self.roaming_list = []
        
        # self.setup_npc(self.vehicle_num, self.ped_num)
        self.roamintg = None
        self.reset()

        print("world agent initialized!")

    def find_recording_dir(self, root_dir):
        record_dir = os.path.join(root_dir, str(self.episode))
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        self.args.record_dir = record_dir

    def store_guide(self, guide, p):
        self.guide_q.append(guide)
        self.guide_dis_q.append(p)
    
    def setup_player(self):
        # set up a vehicle player as the main agent 
        blueprint_library = self.world.get_blueprint_library()
        vehicles = self.world.get_blueprint_library().filter('vehicle.ford.mustang')
        vehicle_bp = vehicles[0]
        player_spawn_point = None
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            chosen_points = spawn_points[29:33] + spawn_points[34:44] + spawn_points[45: 55] + spawn_points[56:58] + spawn_points[87: 97] + spawn_points[98: 108] + spawn_points[109: 119] + spawn_points[120: 122]
            player_spawn_point = random.choice(chosen_points) 
            self.player = self.world.try_spawn_actor(vehicle_bp, player_spawn_point) 

        # setup the npc vehicles
        while len(self.npc_list) < self.args.vehicle_num:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            while spawn_point == player_spawn_point:
                spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(
                        random.choice(blueprint_library.filter('vehicle.*')),
                        spawn_point)
            if vehicle is not None:
                self.npc_list.append(vehicle) 
                self.npc_list[-1].set_autopilot()

    def setup_sensor(self):
        # set up the camera to observe surrounding
        self.rgb_camera = CameraManager(self.player, self.args, tag="RGB")
        self.seg_camera = CameraManager(self.player, self.args, tag="SEG")
        # self.depth_camera = CameraManager(self.player,
        # self.args, tag="DEP")
        self.rgb_camera.set_sensor(RGB_CAMERA_INDEX)
        self.seg_camera.set_sensor(SEG_CAMERA_INDEX)
        # self.depth_camera.set_sensor(1)
        self.col_sensor = CollisionSensor(self.player)
        self.offroad_detector = LaneInvasionSensor(self.player)

        # monitoring the vehicle driving
        self.monitor = CameraManager(self.player, self.args, x=-5.5, y=0, z=2.8, monitor=True, tag="MONITOR")
        self.monitor.set_sensor(0)
    
    def info(self):
        v = self.player.get_velocity()
        self.collision = self.col_sensor.check()
        self.offlane, self.offroad  = False, False

        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        location = self.player.get_location()
        waypoint = self.map.get_waypoint(location, project_to_road=False)
        if waypoint is not None:
            # use Cosine theorem to judge whether the agent is driving on the wrong lanes
            v1 = waypoint.transform.get_forward_vector()
            v2 = self.player.get_velocity()
            v1 = np.array([v1.x, v1.y])
            v2 = np.array([v2.x, v2.y])
            Lx = np.sqrt(v1.dot(v1))
            Ly = np.sqrt(v2.dot(v2))
            if Ly > 0.00001:
                cos_angle = v1.dot(v2) / (Lx*Ly)
                if cos_angle < 0:
                    self.offlane = True
            if waypoint.lane_type != "driving" and Ly > 0.00001:
                self.offroad = True
        else:
            self.offroad = True
        info = {}
        info["speed"] = speed
        info["collision"] = 1 if (self.collision or (self.collision_cnt > 0 and speed < 0.5)) else 0 
        info["offlane"] = 1 if self.offlane else 0
        # Only in carla later than 0.9.4, the python api to return the lane
        # type is provided, which can be used to detect offroad.
        info["offroad"] = 1 if self.offroad else 0
        info["expert_control"] = self.args.autopilot
        coarse_bboxes = self.seg_camera.bbox()
        bbox_3d_list, rotations = ClientSideBoundingBoxes.get_bounding_boxes(self.npc_list, self.rgb_camera.sensor)
        self.bbox_3d_buffer.append(copy.deepcopy(bbox_3d_list))
        self.rotations_buffer.append(rotations)
        if len(self.rotations_buffer) < 3:
            visible_2d_bbox, visible_rotations, all_bboxes= crop_visible_bboxes(coarse_bboxes, self.bbox_3d_buffer[-len(self.rotations_buffer)], self.rotations_buffer[-len(self.rotations_buffer)], self.args.frame_width, self.args.frame_height)
        else:
            visible_2d_bbox, visible_rotations, all_bboxes= crop_visible_bboxes(coarse_bboxes, self.bbox_3d_buffer[-3], self.rotations_buffer[-3], self.args.frame_width, self.args.frame_height)
        self_rotation = self.player.get_transform().rotation
        self_rotation = np.array([self_rotation.yaw, self_rotation.roll, self_rotation.pitch])
        info["bbox"] = visible_2d_bbox
        info["all_bbox"] = all_bboxes
        info['seg_bbox'] = coarse_bboxes
        if len(visible_rotations) == 0:
            info["orientation"] = []
        else:
            info["orientation"] = visible_rotations - self_rotation
        info["seg"] = self.seg_camera.observe()

        return info
    
    def observe(self):
        return self.rgb_camera.observe()
    
    '''
    def record_video(self, tag):
        print("produce {} video of episode: {}".format(tag, self.episode))
        assert(tag in ["seg", "mon", "obs"])
        img_dir = os.path.join(self.args.record_dir, tag)
        if tag == "mon":
            target_q = self.mon_q
        elif tag == "seg":
            target_q = self.seg_q
        elif tag == "obs":
            target_q = self.obs_q

        assert(len(self.timestamp_q) == len(target_q) == len(self.all_bbox_q) == len(self.guide_q) + 1)
        video_len = len(self.timestamp_q)

        for i in range(video_len-1):
            target_frame = target_q[i]
            if tag == "mon":
                img = monitor_guide(target_frame, self.guide_q[i], self.guide_dis_q[i])
                img = img / 255
                matplotlib.image.imsave(os.path.join(img_dir, "%08d.png" % self.timestamp_q[i]), img)
            elif tag == "obs":
                bboxes = self.bbox_q[i]
                seg_bbox = self.seg_bbox_q[i]
                for bbox in bboxes:
                    minr, minc, maxr, maxc = bbox
                    cv2.rectangle(target_frame, (max(minr-2, 0), max(minc-2, 0)), (maxr+2, maxc+2), (0, 255, 60), thickness=1)
                bboxes_all = self.all_bbox_q[i]
                for bbox in bboxes_all:
                    minr, minc, maxr, maxc = bbox
                    cv2.rectangle(target_frame, (minr, minc), (maxr, maxc), (0, 50, 255), thickness=1)
                for bbox in seg_bbox:
                    minc, minr, maxc, maxr = bbox
                    cv2.rectangle(target_frame, (minr, minc), (maxr, maxc), (255, 0, 0), thickness=1)
                matplotlib.image.imsave(os.path.join(img_dir, "%08d.png" % self.timestamp_q[i]), target_frame)
            else:
                matplotlib.image.imsave(os.path.join(img_dir, "%08d.png" % self.timestamp_q[i]), target_frame)

        fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
        img_list = os.listdir(img_dir)
        num = self.timestamp_q[-1]
        video_dir = os.path.join(self.args.record_dir, "{}_{}.avi".format(tag, num))
        vw = cv2.VideoWriter(video_dir, fourcc, 24, \
            (self.args.frame_width, self.args.frame_height))
        for i in range(1, self.timestamp_q[-1] + 1):
            try:
                frame = cv2.imread(os.path.join(img_dir, "%08d.png" % i))
            except:
                continue
            vw.write(frame)
        vw.release()
        # shutil.rmtree(img_dir)
        '''

    def reset(self, testing=False):
        print('Starting new episode ...')
        if (self.args.monitor or self.args.recording_frame) and self.episode > 1:
            time.sleep(2)
            # if self.args.recording_frame:
            #     self.record_video("obs")
            #     self.record_video("seg")
            # if self.args.monitor:
            #     self.record_video("mon")

        self.testing = testing
        self.episode += 1
        self.crosssolid_cnt = 0
        self.timestamp = 0
        self.collision = 0
        self.stuck_cnt = 0
        self.collision_cnt = 0
        self.offroad_cnt = 0
        self.ignite = False
        # self.obs_q = []
        # self.seg_q = []
        # self.mon_q = []
        self.guide_q = []
        self.guide_dis_q = []
        self.timestamp_q = []
        self.bbox_q = []
        self.all_bbox_q = []
        self.seg_bbox_q = []
        self.bbox_3d_buffer = []
        self.rotations_buffer = []

        # set all traffic lights always green
        for actor in self.world.get_actors():
            if actor.type_id == "traffic.traffic_light":
                actor.set_green_time(99999999999999999999)
                actor.set_state(carla.TrafficLightState.Green)

        actors = self.world.get_actors()
        self.log.flush()
        self.log.write("\n\n=============== episode: {} =============== \n\n".format(self.episode))

        # set the director for recording of each frame
        if self.args.recording_frame or self.args.monitor:
            self.find_recording_dir(os.path.join(self.args.save_path, self.args.monitor_video_dir))

        # set directories for realtime frame recording if needed
        if self.args.recording_frame:
            if not os.path.isdir(os.path.join(self.args.record_dir, "obs")):
                os.makedirs(os.path.join(self.args.record_dir, "obs"))
            if not os.path.isdir(os.path.join(self.args.record_dir, "seg")):
                os.makedirs(os.path.join(self.args.record_dir, "seg"))

        if self.args.monitor:
            if not os.path.isdir(os.path.join(self.args.record_dir, "mon")):
                os.makedirs(os.path.join(self.args.record_dir, "mon"))

        if self.player is not None:
            self.destroy()
            self.player = None
        
        # setup the POV player and bind sensor/cameras
        while self.player is None:
            self.setup_player()
            self.setup_sensor()

        time.sleep(4.5) # the world needs some time to spawn actors and sensors
        print("===========================")
        print("reset finished, in total {} actors in world".format(len(self.world.get_actors())))
        print("===========================")
        # self.player.set_velocity(1)
        for frame in range(30):
            self.tick()
            # self.timestamp += 1
            # self.record()
            control = self.player.get_control()
            control.steer = random.uniform(-1.0, 1.0)
            control.throttle = 0.6
            control.brake = 0.0
            hand_brak = False
            reverse = False
            self.player.apply_control(control)

        for i in range(2):
            info = self.info()
            self.tick()

        time1 = time.time()
        info = self.info()
        time2 = time.time()
        obs = self.observe()
        time3 = time.time()
        print("used {}s to get info; {}s to get observation".format(time2-time1, time3-time2))
        mon = self.monitor.observe()
        self.log.write("step {}: speed: {} | collision: {} | offroad: {} | offlane: {}\n".format(self.timestamp, round(info['speed'], 5), info['collision'], info['offroad'], info['offlane']))
        self.timestamp_q.append(self.timestamp)
        
        # save sensor snapshots
        self.record(obs, labels_to_segimage(info['seg']), mon, info)
        # self.obs_q.append(copy.deepcopy(obs))
        # self.seg_q.append(copy.deepcopy(info["seg"]))
        # self.mon_q.append(copy.deepcopy(mon))

        self.bbox_q.append(copy.deepcopy(info["bbox"]))
        self.all_bbox_q.append(copy.deepcopy(info['all_bbox']))
        self.seg_bbox_q.append(copy.deepcopy(info["seg_bbox"]))
        return obs, info

    def record(self, obs, seg, mon, info=dict()):
        width = self.view_w
        height = self.view_h
        obs = obs.astype(np.int32)
        if self.save_record: 
            self.mm.save("mon", self.episode, self.timestep, mon, info)
            self.mm.draw_and_save("obs", self.episode, self.timestep, obs, info)
            self.mm.save("seg", self.episode, self.timestep, seg)

    def step(self, action, expert=False):
        # 1. set and send control signal
        throttle, steer = action
        reverse = False
        brake = 0
        hand_brake = False
        control = self.player.get_control()
        self.timestamp += 1
        self.tick()
        if expert:
            control.throttle = throttle
            control.reverse = reverse
            control.steer = steer
            control.brake = brake
            control.hand_brake = hand_brake
            self.player.apply_control(control)
        else:
            control.throttle = throttle*0.4 + 0.6
            control.steer = steer*0.5
            control.brake = 0
            control.hand_brake = False
            control.reverse = False
            self.player.apply_control(control)

        obs = self.observe()
        info = self.info()
        mon = self.monitor.observe()
        self.timestamp_q.append(self.timestamp)
        self.record(obs, labels_to_segimage(info['seg']), mon, info)
        # self.obs_q.append(copy.deepcopy(obs))
        # self.seg_q.append(copy.deepcopy(info["seg"]))
        # self.mon_q.append(copy.deepcopy(mon))
        self.bbox_q.append(copy.deepcopy(info["bbox"]))
        self.all_bbox_q.append(copy.deepcopy(info['all_bbox']))
        self.seg_bbox_q.append(copy.deepcopy(info["seg_bbox"]))

        reward = self.reward_from_info(info)
        done = self.done_from_info(info) or self.timestamp > 1000
        self.log.write("step {}: speed: {} | collision: {} | offroad: {} | offlane: {}\n".format(self.timestamp, round(info['speed'], 5), info['collision'], info['offroad'], info['offlane']))
        if done:
            if self.stuck_cnt > 30 and self.timestamp > 50:
                self.log.write("done because of stuck\n")
            if self.offroad_cnt > 30:
                self.log.write("done because of offroad\n")
            if self.collision_cnt > 20:
                self.log.write("done because of collision\n")
            if self.timestamp > 1000:
                self.log.write("done because of timestamp out\n")
            if self.save_record:
                for subdir in ['mon', 'seg', 'obs']:
                    self.mm.merge(subdir, self.episode, self.timestamp+1)
        
        return obs, reward, done, info, self.timestamp

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock=None):
        self.world.tick()
        try:
            # this only works in sync mode
            self.world.wait_for_tick()
        except:
            pass

    def render(self, display):
        self.camera_manager.render(display)

    def reward_from_info(self, info):
        reward = dict()
        reward['without_pos'] = info['speed'] / 10 - info['offroad'] - info['collision'] * 2.0
        reward['with_pos'] = reward['without_pos'] - info['offlane'] / 5
        return reward['with_pos']

    def done_from_info(self, info):
        if info['collision'] > 0 or (self.collision_cnt > 0 and info['speed'] < 0.5):
            self.collision_cnt += 1
        else:
            self.collision_cnt = 0

        self.ignite = self.ignite or info['speed'] > 1
        stuck = int(info['speed'] < 1)
        self.stuck_cnt = (self.stuck_cnt + stuck) * stuck * int(bool(self.ignite) or self.testing)

        if info['offroad'] == 1:
            self.offroad_cnt += 1
        return (self.stuck_cnt > 30 and self.timestamp > 50) or self.collision_cnt > 20 or self.offroad_cnt > 30

    def destroySensors(self):
        self.rgb_camera.sensor.destroy()
        self.rgb_camera = None
        self.seg_camera.sensor.destroy()
        self.seg_camera = None
        self.monitor.sensor.destroy()
        self.monitor = None

    def destroy(self):
        actors = [
            self.rgb_camera.sensor,
            self.seg_camera.sensor,
            self.col_sensor.sensor,
            self.offroad_detector.sensor,
            self.monitor.sensor,
            self.player] + self.npc_list
        print("==========================")
        print("destory {} actors in the world".format(len(actors)))
        print("==========================")
        for actor in actors:
            if actor is not None:
                actor.destroy()
                actor = None
        self.npc_list = []
        
