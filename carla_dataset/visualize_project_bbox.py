import cv2 
import numpy as np  
import os
import numpy 
import math
from math import *
import copy
from numpy.linalg import inv

data_dir = "v4_demo"
output_dir = "vis_compare"
obs_dir = "{}/obs".format(data_dir)
orientation_dir = "{}/orientations".format(data_dir)
bbox_dir = "{}/3d_bbox".format(data_dir)
dimensions_dir = "{}/dimensions".format(data_dir)
depth_dir = "{}/depth".format(data_dir)
calib_dir = "{}/calib".format(data_dir)

if not os.path.isdir(output_dir):
    os.makedirs("{}/original".format(output_dir))
    os.makedirs("{}/recovered".format(output_dir))
    os.makedirs("{}/concat".format(output_dir))

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
    

def vertex_3d_to_2d(pos_vector, intrinsic, extrinsic):
    # transform a 3D vertex coordinate to 2d observation plane

    transformed_3d_pos = np.dot(inv(extrinsic), pos_vector)
    pos2d = np.dot(intrinsic, transformed_3d_pos[:3]) # 4th of inv(extrinsic.matrix) is 1
    pos2d = np.array([
        pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]
    ])
    return pos2d


def vertex_2d_to_3d(pos2d, width, height, instrinsic, extrinsic):
    # reverse of the function vertex_2d_to_3d
    # pos2d, instrinsic, extrinsic should all be np.array
    relative_x = pos2d[0]
    relative_y = pos2d[1]
    pos2d_r = np.array([relative_x * pos2d[2], relative_y * pos2d[2], pos2d[2]])
    transformed_3d_pos = np.ones([4,1])
    transformed_3d_pos[:3] = np.dot(np.linalg.inv(instrinsic), pos2d_r)
    pos_vector = np.dot(extrinsic, transformed_3d_pos)
    return pos_vector, transformed_3d_pos


def draw_3d_bbox(img, bbox3d, tnk=1, color=(0,0,255)):
    # img: a 2d image array
    # bbox3d: a matrix containing eight vertices to form a 3d bounding box
    # import pdb; pdb.set_trace()
    front_mask = []
    point_pairs = [[0,1], [0,2], [1,3], [2,3],
                   [4,5], [4,6], [5,7], [6,7],
                   [0,4], [1,5], [2,6], [3,7],
                   [1,4], [0,5], [2,7], [3,6]]

    h, w = img.shape[0], img.shape[1]
    
    in_view = 0

    try:
        for point in bbox3d:
            x, y = point
            if (0 < x < w) and (0 < y < h):
                in_view += 1
    except:
        return img

    if in_view < 3:
        return img

    for pair in point_pairs:
        ind1, ind2 = pair[0], pair[1]
        try:
            p1, p2 = bbox3d[ind1], bbox3d[ind2]
        except:
            continue
        point1 = (p1[0], p1[1])
        point2 = (p2[0], p2[1])
        
        img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, tnk)

    return img


def main1():
    obs_list = os.listdir(obs_dir)
    frame_num = len(obs_list)
    for ind in range(frame_num):
        find = ind + 1
        fobs = "{}/{}.jpg".format(obs_dir, find)
        fori = "{}/{}.npy".format(orientation_dir, find)
        fbbox = "{}/{}.npy".format(bbox_dir, find)
        fdepth = "{}/{}.npy".format(depth_dir, find)
        fdim = "{}/{}.npy".format(dimensions_dir, find)
        im = cv2.imread(fobs)
        orientations = np.load(fori)
        bboxs = np.load(fbbox)
        depth = np.load(fdepth)
        dims = np.load(fdim)
        bbox_num = min(bboxs.shape[0], orientations.shape[0])
        for bboxind in range(bbox_num):
            # print(bbox_num, orientations.shape)
            # print(find)
            bbox = bboxs[bboxind]
            ori = orientations[bboxind]
            dim = dims[bboxind]
            bbox = bbox.astype(int)
            bbox = bbox[:,:2]
            im = draw_3d_bbox(im, bbox)
            center = (int(round(bbox[:,0].sum()/8)), int(round(bbox[:,1].sum()/8)))
            if 0 < center[0] < 512 and 0 < center[1] < 256:
                cv2.circle(im, center, radius=2, color=(0, 255, 255), thickness=2)
                pointdepth = round(round(depth[center[1]][center[0]], 3) * 100, 2)
                cv2.putText(im, "{}".format(pointdepth), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
        
        cv2.imwrite("vis_1/{}.jpg".format(find), im)


def recover_3d_bbox(center_point, dx, dy, dz, rot):
    # recover the 3d bbox from center point position
    # and dimensions, rotation towards the world yaw axes
    dx1 = dx*cos(rot) + dy*sin(rot)
    dy1 = dx*sin(rot) - dy*cos(rot)
    dx2 = dx*cos(rot) - dy*sin(rot)
    dy2 = dx*sin(rot) + dy*cos(rot)
    dx3 = -dx1
    dy3 = -dy1 
    dx4 = -dx2 
    dy4 = -dy2

    dpoint1 = [dx4, dy4, dz]
    dpoint2 = [dx1, dy1, dz]
    dpoint3 = [dx3, dy3, dz]
    dpoint4 = [dx2, dy2, dz]
    dpoint5 = [dx4, dy4, -dz]
    dpoint6 = [dx1, dy1, -dz]
    dpoint7 = [dx3, dy3, -dz]
    dpoint8 = [dx2, dy2, -dz]

    bbox3d = [center_point+dpoint1, center_point+dpoint2,
                center_point+dpoint3, center_point+dpoint4,
                center_point+dpoint5, center_point+dpoint6,
                center_point+dpoint7, center_point+dpoint8]

    bbox3d = np.array(bbox3d)

    return bbox3d


def main():
    obs_list = os.listdir(obs_dir)
    frame_num = len(obs_list)
    for ind in range(frame_num):
        find = ind + 1
        fobs = "{}/{}.jpg".format(obs_dir, find)
        fori = "{}/{}.npy".format(orientation_dir, find)
        fbbox = "{}/{}.npy".format(bbox_dir, find)
        fdepth = "{}/{}.npy".format(depth_dir, find)
        fdim = "{}/{}.npy".format(dimensions_dir, find)
        fext = "{}/extrinsic_{}.npy".format(calib_dir, find)
        fins = "{}/intrinsic_{}.npy".format(calib_dir, find)
        im = cv2.imread(fobs)
        orientations = np.load(fori)
        bboxs = np.load(fbbox)
        depth = np.load(fdepth)
        dims = np.load(fdim)
        extrinsic = np.load(fext)
        R = extrinsic[:3, :3]
        eulers = np.array(euler_from_matrix(R))
        ax, ay, az = eulers / math.pi * 180

        instrinsic = np.load(fins)
        bbox_num = min(bboxs.shape[0], orientations.shape[0])
        for bboxind in range(bbox_num):
            # print(bbox_num, orientations.shape)
            # print(find)
            dx, dy, dz = dims[bboxind][0], dims[bboxind][1], dims[bboxind][2]
            bbox = bboxs[bboxind]
            # print(bbox)
            orien = orientations[bboxind]
            bpoint_list = []
            pos_vector_list = []
            transformed_3d_pos_list = []
            for bpoint in bbox:
                bpoint = np.expand_dims(bpoint, axis=1)
                print("****")
                print(bpoint)
                pos_vector, transformed_3d_pos = vertex_2d_to_3d(bpoint, 512, 256, instrinsic, extrinsic)
                print("####")
                print(bpoint)
                bpoint_ = vertex_3d_to_2d(pos_vector, instrinsic, extrinsic)
                print("-----")
                print(bpoint)
                print(bpoint_)
                print(bpoint == bpoint_)
                print("-----")
                pos_vector_list.append(pos_vector)
                bpoint_list.append(bpoint_)
                transformed_3d_pos_list.append(transformed_3d_pos)

            bpoints = np.array(bpoint_list).squeeze(2)
            pos_vectors = np.array(pos_vector_list).squeeze(2)
            transformed_3d_poses = np.array(transformed_3d_pos_list)
            center_point = np.mean(pos_vectors, axis=0)[:3]
            rot = (orien + ax) * 2 * math.pi / 360.0
            gt_3d_bbox = pos_vectors[:,:3]
            recovered_3d_bbox = recover_3d_bbox(center_point, dx, dy, dz, rot)
            recovered_2d_corners = []
            for corner3d in recovered_3d_bbox:
                pos_vector = np.concatenate([corner3d, np.array([1.])])
                bpoint_ = vertex_3d_to_2d(pos_vector, instrinsic, extrinsic)
                recovered_2d_corners.append(bpoint_)
            recovered_2d_corners = np.array(recovered_2d_corners)

            im2 = copy.deepcopy(im)
            # draw original 3d bbox
            bbox = bbox.astype(int)
            bbox = bbox[:,:2]
            im = draw_3d_bbox(im, bbox)
            center = (int(round(bbox[:,0].sum()/8)), int(round(bbox[:,1].sum()/8)))
            if 0 < center[0] < 512 and 0 < center[1] < 256:
                cv2.circle(im, center, radius=2, color=(0, 255, 255), thickness=2)
                pointdepth = round(round(depth[center[1]][center[0]], 3) * 100, 2)
                cv2.putText(im, "{}".format(pointdepth), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
            

            # draw recovered 3d bbox
            recovered_2d_corners = recovered_2d_corners.astype(int)
            recovered_2d_corners = recovered_2d_corners[:,:2]
            im2 = draw_3d_bbox(im2, recovered_2d_corners)
            center = (int(round(recovered_2d_corners[:,0].sum()/8)), int(round(recovered_2d_corners[:,1].sum()/8)))
            if 0 < center[0] < 512 and 0 < center[1] < 256:
                cv2.circle(im2, center, radius=2, color=(0, 255, 255), thickness=2)
                pointdepth = round(round(depth[center[1]][center[0]], 3) * 100, 2)
                cv2.putText(im2, "{}".format(pointdepth), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
            
            im3=np.concatenate([im, im2], axis=1)
            
        
        cv2.imwrite("{}/original/{}.jpg".format(output_dir, find), im)
        cv2.imwrite("{}/recovered/{}.jpg".format(output_dir, find), im2)
        cv2.imwrite("{}/concat/{}.jpg".format(output_dir, find), im3)
    

if __name__ == "__main__":
    main()