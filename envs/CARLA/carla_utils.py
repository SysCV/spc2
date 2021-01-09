import cv2
import os
from .carla_lib.carla.transform import Transform
import numpy as np
from .carla_lib.carla.settings import CarlaSettings
from .carla_lib.carla.sensor import Camera
# from skimage import measure
import math
from numpy.linalg import inv
import numpy

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print(axes, "failed")
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def tighten_bbox(bboxes, bboxes3d, seg_bboxes, width, height):
    # tighten the bbox boundary referring to segmentation carving
    # bboxes: come from 3D bbox, whose boundaries are loose
    # seg_bboxes: come from semantic segmentation carving, which might contatins multiple instance
    assert(len(bboxes) == len(bboxes3d))
    tight_bboxes = []
    visible_bbox_indices = []
    visible_bboxes3d = []
    for ind in range(len(bboxes)):
        bbox = bboxes[ind]
        xb1, yb1, xb2, yb2 = bbox
        xmin, ymin, xmax, ymax = width-1, height-1, 0, 0
        has_seg_bbox_inside = False
        for seg_bbox in seg_bboxes:
            xs1, ys1, xs2, ys2 = seg_bbox
            if xs1 >= xb2 or xs2 <= xb1 or ys1 >= yb2 or ys2 <= yb1:
                # no overlap
                continue
            else:
                has_seg_bbox_inside = True
                xmin = min(xmin, xs1)
                ymin = min(ymin, ys1)
                xmax = max(xmax, xs2)
                ymax = max(ymax, ys2)
        # union of seg-bboxes which have overlap with bbox
        # assert(has_seg_bbox_inside) # at lease one seg bbox should be inside the loose bbox
        if not has_seg_bbox_inside:
            continue
        else:
            # there is vehicle semantic area in this bbox
            xb1 = max(xmin, xb1)
            yb1 = max(ymin, yb1)
            xb2 = min(xmax, xb2)
            yb2 = min(ymax, yb2)
            if (xb2 - xb1) * (yb2 - yb1) > 8:
                # ignore the vehicle instances that are too tiny
                # maintain the same number of 3d and 2d bboxes
                visible_bboxes3d.append(bboxes3d[ind])
                tight_bboxes.append([xb1, yb1, xb2, yb2])
                visible_bbox_indices.append(ind)
    return tight_bboxes, visible_bbox_indices, visible_bboxes3d


def extract_agent_bbox(agent):
    # extract the spatial information of an agent from CARLA measurements
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

    rotation = agent.vehicle.transform.rotation
    vehicle_location = agent.vehicle.transform.location
    dimensions = [ext.x, ext.y, ext.z]
    
    return bbox, rotation, vehicle_location, dimensions


def vertex_3d_to_2d(vertex, intrinsic, extrinsic):
    # transform a 3D vertex coordinate to 2d observation plane
    pos_vector = np.array([
        [vertex[0,0]],  # [[X,
        [vertex[0,1]],  #   Y,
        [vertex[0,2]],  #   Z,
        [1.0]           #   1.0]]
    ])

    transformed_3d_pos = np.dot(inv(extrinsic.matrix), pos_vector)
    pos2d = np.dot(intrinsic, transformed_3d_pos[:3]) # 4th of inv(extrinsic.matrix) is 1
    pos2d = np.array([
        pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]
    ])
    return pos2d

def vertex_2d_to_3d(pos2d, width, height, instrinsic, extrinsic):
    # reverse of the function vertex_2d_to_3d
    # pos2d, instrinsic, extrinsic should all be np.array
    pos2d[0] = width - pos2d[0]
    pos2d[1] = height - pos2d[1]
    pos2d = np.array([pos2d[0] * pos2d[2], pos2d[1] * pos2d[2], pos2d[2]])
    transformed_3d_pos = np.ones([4,1])
    transformed_3d_pos[:3] = np.dot(np.linalg.inv(instrinsic), pos2d)
    pos_vector = np.dot(extrinsic, transformed_3d_pos)
    return pos_vector


def default_settings(args, h, w):
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=args.vehicle_num,
        NumberOfPedestrians=args.ped_num,
        WeatherId=args.weather_id,  
        PlayerVehicle='/Game/Blueprints/Vehicles/Mustang/Mustang.Mustang_C',
        QualityLevel='Epic')
    settings.randomize_seeds()

    # set rgb camera
    camera_RGB = Camera('CameraRGB')
    camera_RGB.set(FOV=90.0)
    camera_RGB.set_image_size(w, h)
    camera_RGB.set_position(1, 0, 2.50)
    camera_RGB.set_rotation(0, 0, 0)
    settings.add_sensor(camera_RGB)

    # set the rgb monitoring camera behind the player agent
    camera_MON = Camera('CameraMON')
    camera_MON.set_image_size(w, h)
    camera_MON.set_position(-7.0, 0, 2.80)
    settings.add_sensor(camera_MON)

    # set the semantic segmentation camera
    camera_seg = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
    camera_seg.set_image_size(w, h)
    camera_seg.set_position(1, 0, 2.50)
    settings.add_sensor(camera_seg)

    # set the depth camera
    camera_depth = Camera('CameraDepth', PostProcessing='Depth')
    camera_depth.set(FOV=90.0)
    camera_depth.set_image_size(w, h)
    camera_depth.set_position(1, 0, 2.50)
    camera_depth.set_rotation(0, 0, 0)
    settings.add_sensor(camera_depth)

    obs_calibration = np.identity(3)
    obs_calibration[0, 2] = args.frame_width / 2
    obs_calibration[1, 2] = args.frame_height / 2
    obs_calibration[0, 0] = obs_calibration[1, 1] = args.frame_width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    obs_to_car_transform = camera_RGB.get_unreal_transform()

    return settings, obs_calibration, obs_to_car_transform

def seg_to_bbox(seg, vehicle_id=5):
    # generate coarse boundbing box from semantic segmantation map
    vehicle_bboxes = []
    vehicle_seg = measure.label(np.where(seg == vehicle_id, 1, 0))
    for vehicle in measure.regionprops(vehicle_seg):
        if vehicle.area < 20:
            continue
        y1, x1, y2, x2 = vehicle.bbox
        vehicle_bboxes.append([x1, y1, x2, y2])
    return np.array(vehicle_bboxes)


def draw_3d_bbox(img, bbox3d, tnk=1, color=(0,0,255)):
    # img: a 2d image array
    # bbox3d: a matrix containing eight vertices to form a 3d bounding box
    # import pdb; pdb.set_trace()
    front_mask = []
    point_pairs = [[0,1], [0,2], [1,3], [2,3],
                   [4,5], [4,6], [5,7], [6,7],
                   [0,4], [1,5], [2,6], [3,7],
                   [1,4], [0,5], [2,7], [3,6]]
    
    for pair in point_pairs:
        ind1, ind2 = pair[0], pair[1]
        p1, p2 = bbox3d[ind1], bbox3d[ind2]
        img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, tnk)

    return img


def labels_to_segimage(array):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [0, 255, 0],     # Pedestrians
        5: [255, 255, 0],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }
    result =  np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


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
