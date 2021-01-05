import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import skimage
import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 512//2
VIEW_HEIGHT = 256//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

def crop_visible_bboxes(seg_bbox, bboxes_3d_visible, rotations_visible, width, height):
    # generate the real instance bounding boxes 
    rotations_final = []
    bboxes_final = []
    num = len(bboxes_3d_visible)
    all_bboxes = []
    for i in range((num)):
        bbox = bboxes_3d_visible[i]
        # print(bbox)
        x1 = int(np.min(bbox[:, 0]))+ 1
        y1 = int(np.min(bbox[:, 1]))+ 1
        x2 = int(np.max(bbox[:, 0]))
        y2 = int(np.max(bbox[:, 1]))
        if x1 > width or y1 > height or x2 < 0 or y2 < 0:
            continue
        vehicle_width = x2 - x1
        vehicle_height = y2 - y1
        x1 = max(int(x1 - 0.05 * vehicle_width), 0)
        y1 = max(int(y1 - 0.05 * vehicle_height), 0)
        x2 = min(int(x2 + 0.05 * vehicle_width), width)
        y2 = min(int(y2 + 0.05 * vehicle_height), height)
        all_bboxes.append([x1, y1, x2, y2])
        overlap = False
        seg_x1_list = []
        seg_y1_list = []
        seg_x2_list = []
        seg_y2_list = []
        for seg in seg_bbox:
            seg_y1, seg_x1, seg_y2, seg_x2 = seg
            if (x1 <= seg_x1 <= x2 and y1 <= seg_y1 <= y2) and (x1 <= seg_x2 <= x2 and y1 <= seg_y2 <= y2):
                overlap = True
                seg_x1_list.append(max(seg_x1, x1))
                seg_y1_list.append(max(seg_y1, y1))
                seg_x2_list.append(min(seg_x2, x2))
                seg_y2_list.append(min(seg_y2, y2))
        if overlap:
            bboxes_final.append([min(seg_x1_list), min(seg_y1_list), max(seg_x2_list), max(seg_y2_list)])
            # print("output the visible bbox:")
            # print(bbox)
            # print([min(seg_x1_list), min(seg_y1_list), max(seg_x2_list), max(seg_y2_list)])
            rotations_final.append(rotations_visible[i])
    return np.array(bboxes_final), np.array(rotations_final), np.array(all_bboxes)


# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        '''
        bounding_boxes = np.array([ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles])
        # filter objects behind camera
        inscene_indices = np.array([1 if all(bb[:, 2]) > 0 else 0 for bb in bounding_boxes])
        inscene_indices = np.argwhere(inscene_indices == 1)
        bounding_boxes_visible = [bb for ]
        rotations = [vehicle.get_transform().rotation for vehicle in vehicles]
        rotation_decomposed = np.array([[rotation.yaw, rotation.roll, rotation.pitch] for rotation in rotations])
        rotations_visible = rotation_decomposed[inscene_indices]
        return bounding_boxes_visible, rotations_visible
        '''
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        rotations = []
        for vehicle in vehicles:
            transform = vehicle.get_transform()
            rotation = transform.rotation
            rotations.append([rotation.yaw, rotation.roll, rotation.pitch])
        inscene_indices = []
        for i in range(len(bounding_boxes)):
            bb = bounding_boxes[i]
            if all(bb[:, 2] > 0):
                inscene_indices.append(1)
            else:
                inscene_indices.append(0) 
        inscene_boxes = [bounding_boxes[i] for i in range(len(bounding_boxes)) if inscene_indices[i] == 1]
        return inscene_boxes, np.array(rotations)[inscene_indices]

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """
        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix