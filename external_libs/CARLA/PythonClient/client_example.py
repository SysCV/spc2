#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


#### This script has been deprecated !!!! ##### 

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import numpy as np

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.image_converter import labels_to_array
from carla.transform import Transform


WIDTH　=　512
HEIGHT　=　256


def simplify_seg(array):
    classes = {
        0: 0,   # None
        1: 1,   # Buildings
        2: 1,   # Fences
        3: 1,   # Other
        4: 1,   # Pedestrians
        5: 1,   # Poles
        6: 4,   # RoadLines
        7: 3,   # Roads
        8: 2,   # Sidewalks
        9: 1,   # Vegetation
        10: 5,  # Vehicles
        11: 1,  # Walls
        12: 1   # TrafficSigns
    }

    result = np.zeros_like(array, dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def convert_image(sensor_data, width, height, simple_seg=True):
    obs = np.frombuffer(sensor_data['CameraRGB'].raw_data, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    mon = np.frombuffer(sensor_data['CameraMON'].raw_data, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    seg = labels_to_array(sensor_data['CameraSegmentation'])
    if simple_seg:
        seg = simplify_seg(seg)
    return obs, seg, mon


def get_bbox(self, measurement, seg):
    global 
    width = WIDTH
    height = HEIGHT
    extrinsic = Transform(measurement.player_measurements.transform) * self.obs_to_car_transform 
    bbox_list = []
    orientation_list = []
    distance_list = []
    # main_rotation = measurement.player_measurements.transform.rotation
    player_location = measurement.player_measurements.transform.location
    player_location = np.array([player_location.x, player_location.y, player_location.z])
    # collect the 2bbox generated from the 3d-bbox of non-player agents
    for agent in measurement.non_player_agents:
        if agent.HasField("vehicle"):
            # veh_id = agent.id
            # idx = self.nonplayer_ids[veh_id]
            vehicle_transform = Transform(agent.vehicle.transform)
            bbox_transform = Transform(agent.vehicle.bounding_box.transform)
            ext = agent.vehicle.bounding_box.extent
            bbox = np.array([
                [  ext.x,   ext.y,   ext.z],
                [- ext.x,   ext.y,   ext.z],
                [  ext.x, - ext.y,   ext.z],
                [- ext.x, - ext.y,   ext.z],
                [  ext.x,   ext.y, - ext.z],
                [- ext.x,   ext.y, - ext.z],
                [  ext.x, - ext.y, - ext.z],
                [- ext.x, - ext.y, - ext.z]
            ])

            bbox = bbox_transform.transform_points(bbox)
            bbox = vehicle_transform.transform_points(bbox)
            
            orientation = agent.vehicle.transform.orientation
            vehicle_location = agent.vehicle.transform.location
            cur_location = np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z])
            distance = np.linalg.norm(player_location - cur_location)

            vertices = []
            for vertex in bbox:
                pos_vector = np.array([
                    [vertex[0,0]],  # [[X,
                    [vertex[0,1]],  #   Y,
                    [vertex[0,2]],  #   Z,
                    [1.0]           #   1.0]]
                ])
                transformed_3d_pos = np.dot(inv(extrinsic.matrix), pos_vector)
                pos2d = np.dot(self.intrinsic, transformed_3d_pos[:3])
                pos2d = np.array([
                    pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]
                ])
                
                if pos2d[2] > 0:
                    x_2d = width - pos2d[0]
                    y_2d = height - pos2d[1]
                    vertices.append([x_2d, y_2d])
            if len(vertices) > 1:
                # vehicle_rotation = agent.vehicle.transform.rotation
                vertices = np.array(vertices)
                bbox_list.append([np.min(vertices[:, 0]), np.min(vertices[:, 1]),
                    np.max(vertices[:, 0]), np.max(vertices[:, 1])])
                orientation_list.append(orientation)
                distance_list.append(distance)
    seg_bboxes = seg_to_bbox(seg)
    final_bboxes = []
    final_directions = []
    final_distances = []
    assert(len(bbox_list) == len(orientation_list))
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        direction = orientation_list[i]
        xmin, ymin, xmax, ymax = bbox
        x1, y1, x2, y2 = width, height, 0, 0
        for segbbox in seg_bboxes:
            xmin0, ymin0, xmax0, ymax0 = segbbox
            if xmin0 >= xmin - 5 and ymin0 >= ymin - 5 and xmax0 < xmax + 5 and ymax0 < ymax + 5:
                x1 = min(x1, xmin0)
                y1 = min(y1, ymin0)
                x2 = max(x2, xmax0)
                y2 = max(y2, ymax0)
        if x2 > x1 and y2 > y1 and [int(x1), int(y1), int(x2), int(y2)] not in final_bboxes:
            final_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            relative_orientation = get_angle(direction.x, direction.y, self.orientation.x, self.orientation.y)
            final_directions.append(relative_orientation)
            final_distances.append(distance_list[i])
    # for angle in final_directions:
    #     self.angle_logger.write("timestep {}: {}\n".format(self.timestep, angle))
    #     self.angle_logger.flush()
    final_distances = np.array(final_distances)
    visible_coll_num = min(coll_veh_num, final_distances.size)
    coll_idx = np.argpartition(final_distances, visible_coll_num - 1)[:visible_coll_num]
    final_colls = [1 if i in coll_idx else 0 for i in range(final_distances.size)]
    return final_bboxes, final_directions, final_colls

def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    global WIDTH, HEIGHT
    number_of_episodes = 1
    frames_per_episode = 300

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=0,
                NumberOfPedestrians=60,
                WeatherId=1,
                PlayerVehicle='/Game/Blueprints/Vehicles/Mustang/Mustang.Mustang_C',
                QualityLevel='Epic')
            settings.randomize_seeds()

            # Now we want to add a couple of cameras to the player vehicle.
            # We will collect the images produced by these cameras every
            # frame.

            # The default camera captures RGB images of the scene.
            camera0 = Camera('CameraRGB')
            # Set image resolution in pixels.
            camera0.set_image_size(WIDTH, HEIGHT)
            # Set its position relative to the car in meters.
            camera0.set_position(1, 0, 2.50)
            camera0.set_rotation(0, 0, 0)
            settings.add_sensor(camera0)

            # Let's add another camera producing ground-truth depth.
            # camera1 = Camera('CameraDepth', PostProcessing='Depth')
            # camera1.set_image_size(800, 600)
            # camera1.set_position(0.30, 0, 1.30)
            # settings.add_sensor(camera1)

            camera2 = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
            camera2.set_image_size(WIDTH, HEIGHT)
            camera2.set_position(1, 0, 2.50)
            settings.add_sensor(camera2)

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            # number_of_player_starts = len(scene.player_start_spots)
            # player_start = random.randint(0, max(0, number_of_player_starts - 1))

            interval = lambda x, y: list(range(x, y+1))
            player_starts = interval(29, 32) + interval(34, 43) + interval(45, 54) + interval(56, 57) + interval(64, 85) + interval(87, 96) + interval(98, 107) + interval(109, 118) + interval(120, 121)
            player_start = np.random.choice(player_starts)

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()
                obs, seg, mon = convert_image(sensor_data, WIDTH, HEIGHT, True)
                bboxes, directions, _ = get_bbox(measurements, seg)
                # Print some of the measurements.
                # print_measurements(measurements)

                # Save the images to disk if requested.
                if args.save_images_to_disk:
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode, name, frame)
                        measurement.save_to_disk(filename)

                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

                control = measurements.player_measurements.autopilot_control
                control.steer += random.uniform(-0.1, 0.1)
                client.send_control(control)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
