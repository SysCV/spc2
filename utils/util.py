from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle as pkl
import os
import time
from sklearn.metrics import confusion_matrix
import copy
import cv2
import logging
from utils.eval_segm import mean_IU, mean_accuracy, pixel_accuracy, frequency_weighted_IU
import math
import random
import logging
from matplotlib.patches import Wedge


def get_guide_action(action, lb=-1.0, ub=1.0):
        _bin_divide = np.array(self.args.bin_divide)
        
        action = ((action - lb) / (ub - lb) * _bin_divide).astype(np.uint8)
        weight = np.array(list(map(lambda x: np.prod(_bin_divide[:x]), range(len(self.args.bin_divide)))))
        
        return np.sum(action * weight, axis=-1)


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


def setup_logger(logger_name, log_file, level=logging.INFO, resume=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a' if resume else 'w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level) 
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        indices = [e[0] for e in endpoints]
        assert indices == sorted(indices)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value


def generate_guide_grid(bin_divide, lb=-1.0, ub=1.0):
    grids = np.meshgrid(*map(lambda x: (np.arange(x) + 0.5) / x * (ub - lb) + lb, bin_divide))
    return np.concatenate(list(map(lambda x: x.reshape(-1, 1), grids)), axis=-1)


def softmax(x, axis=1):
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis=axis))
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)


def log_frame(obs, action, video_folder, video=None):
    if video is not None:
        video.write(obs)
    with open(os.path.join(video_folder, 'actions.txt'), 'a') as f:
        f.write('time %0.2f action %0.4f %0.4f\n' % (
            time.time(),
            action[0],
            action[1]
        ))


def color_text(text, color):
    color = color.lower()
    if color == 'red':
        prefix = '\033[1;31m'
    elif color == 'green':
        prefix = '\033[1;32m'
    return prefix + text + '\033[0m'


def visualize_guide_action(args, data, outputs, guides, label):
    if not os.path.isdir('visualize/guidance'):
        os.makedirs('visualize/guidance')
    _outputs = F.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    label = label.data.cpu().numpy()
    for i in range(data.shape[0]):
        obs = data[i].data.cpu().numpy().transpose(1, 2, 0)
        obs = cv2.cvtColor((obs * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        action = guides[int(outputs[i])]
        obs = draw_guide(args, obs, 150, 66, 45, _outputs[i].data.cpu().numpy().reshape(-1))
        obs = draw_action(obs, 150, 66, 45, 1, np.array(action))
        gt_action = guides[int(label[i])]
        obs = draw_action(obs, 150, 190, 45, 1, np.array(gt_action))
        cv2.imwrite(os.path.join('visualize', 'guidance', 'guidance_%d.png' % i), obs)


def draw_guide(args, fig, x, y, l, p):
    square = np.ones((args.bin_divide[0]*6+1, args.bin_divide[1]*6+1, 3), dtype=np.uint8) * 128
    p = p * 255 * 10
    for i in range(args.bin_divide[1]):
        for j in range(args.bin_divide[0]):
            square[i*6+1:i*6+6, j*6+1:j*6+6, :] = p[j*args.bin_divide[1]+i]
    square = np.flip(square, axis=0)
    square = cv2.resize(square, (2*l, 2*l))
    fig[x-l:x+l, y-l:y+l, :] = square
    return fig


def draw_action(fig, x, y, l, w, action):
    fig[x-l:x+l, y-w:y+w] = 0
    fig[x-w:x+w, y-l:y+l] = 0
    t = int(abs(action[0]) * l)
    if action[0] > 0:
        fig[x-t:x, y-3*w:y+3*w] = np.array([36, 28, 237])
    else:
        fig[x:x+t, y-3*w:y+3*w] = np.array([36, 28, 237])
    t = int(abs(action[1]) * l)
    if action[1] > 0:
        fig[x-3*w:x+3*w, y:y+t] = np.array([14, 201, 255])
    else:
        fig[x-3*w:x+3*w, y-t:y] = np.array([14, 201, 255])
    return fig


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


def write_log(args, file, func, output, target):
    res = []
    for i in range(args.pred_step + 1):
        tmp = 0
        for j in range(args.batch_size):
            tmp += func(output[j, i, ...], target[j, i, ...])
        res.append(tmp * 100 / args.batch_size)
    with open(file, 'a') as f:
        for i in range(args.pred_step + 1):
            f.write('%0.3f ' % res[i])
        f.write('\n')

def log_seg(args, seg_np, target_np):
    write_log(args, os.path.join(args.save_path, 'mean_IU.txt'), mean_IU, seg_np, target_np)
    write_log(args, os.path.join(args.save_path, 'mean_acc.txt'), mean_accuracy, seg_np, target_np)
    write_log(args, os.path.join(args.save_path, 'pixel_acc.txt'), pixel_accuracy, seg_np, target_np)
    write_log(args, os.path.join(args.save_path, 'freq_IU.txt'), frequency_weighted_IU, seg_np, target_np)


def draw_from_pred_torcs(pred):
    illustration = np.zeros((256, 256, 3)).astype(np.uint8)
    illustration[:, :, 0] = 255
    illustration[pred == 1] = np.array([0, 255, 0])
    illustration[pred == 2] = np.array([0, 0, 0])
    illustration[pred == 3] = np.array([0, 0, 255])
    return illustration


def draw_from_pred_carla(array):
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [255, 0, 0],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }

    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def draw_from_pred_gta(array):
    classes = {
        0: [0, 0, 0],
        1: [255, 255, 255],
        2: [255, 0, 0],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [255, 255, 0],
        6: [0, 255, 255],
        7: [255, 0, 255],
        8: [192, 192, 192],
        9: [128, 128, 128],
        10: [128, 0, 0],
        11: [128, 128, 0],
        12: [0, 128, 0],
        13: [128, 0, 128],
        14: [0, 128, 128],
        15: [0, 0, 128],
        16: [139, 0, 0],
        17: [165, 42, 42],
        18: [178, 34, 34]
    }

    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def draw_from_pred(args, array):
    if 'torcs' in args.env :
        return draw_from_pred_torcs(array)
    elif 'carla' in args.env:
        return draw_from_pred_carla(array)
    elif 'gta' in args.env:
        return draw_from_pred_gta(array)


def visualize(args, target, output):
    if not os.path.isdir('visualize'):
        os.mkdir('visualize')

    batch_id = np.random.randint(args.batch_size)
    observation = (from_variable_to_numpy(target['obs_batch'][batch_id, :, -3:, :, :]) * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
    target['seg_batch'] = target['seg_batch'].view(args.batch_size, args.pred_step + 1, args.frame_height, args.frame_width)
    segmentation = from_variable_to_numpy(target['seg_batch'][batch_id])
    output['seg_pred'] = output['seg_pred'].view(args.batch_size, args.pred_step + 1, args.classes, args.frame_height, args.frame_width)
    _, prediction = torch.max(output['seg_pred'][batch_id], 1)
    prediction = from_variable_to_numpy(prediction)
    for i in range(args.pred_step):
        import pdb; pdb.set_trace()
        cv2.imwrite('visualize/%d.png' % i, np.concatenate([cv2.cvtColor(observation[i], cv2.COLOR_RGB2BGR), draw_from_pred(args, segmentation[i]), draw_from_pred(args, prediction[i])], 1))

    with open(os.path.join(args.save_path, 'report.txt'), 'a') as f:
        if args.use_collision:
            f.write('target collision:\n')
            f.write(str(from_variable_to_numpy(target['coll_batch'][batch_id])) + '\n')
            f.write('output collision:\n')
            f.write(str(from_variable_to_numpy(output['coll_prob'][batch_id])) + '\n')

        if args.use_offroad:
            f.write('target offroad:\n')
            f.write(str(from_variable_to_numpy(target['offroad_batch'][batch_id])) + '\n')
            f.write('output offroad:\n')
            f.write(str(from_variable_to_numpy(output['offroad_prob'][batch_id])) + '\n')

        if args.use_offlane:
            f.write('target offlane:\n')
            f.write(str(from_variable_to_numpy(target['offlane_batch'][batch_id])) + '\n')
            f.write('output offlane:\n')
            f.write(str(from_variable_to_numpy(output['offlane_prob'][batch_id])) + '\n')

        if args.use_speed:
            f.write('target speed:\n')
            f.write(str(from_variable_to_numpy(target['sp_batch'][batch_id, :-1])) + '\n')
            f.write('output speed:\n')
            f.write(str(from_variable_to_numpy(output['speed'][batch_id])) + '\n')


def init_dirs(dir_list):
    for path in dir_list:
        make_dir(path)


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def load_model(args, path, net, resume=True):
    if resume:
        if args.checkpoint != "" and args.eval:
            model_path = args.checkpoint
            state_dict = torch.load(model_path)
            net.load_state_dict(state_dict)
            print("eval | loaded model: {}".format(args.checkpoint))
            epoch = 0
        else:
            model_path = os.path.join(path, 'model')
            if os.path.isdir(model_path):
                file_list = sorted(os.listdir(model_path))
            else:
                os.makedirs(model_path)
                file_list = []
            if len(file_list) == 0 and not args.eval:
                print('No model to resume!')
                model_path = args.pretrain_model
                state_dict = torch.load(model_path)
                print('turn to the base pretrain model: {}'.format(args.pretrain_model))
                net.load_state_dict(state_dict)
                epoch = 0
            else:
                model_path = file_list[-1]
                epoch = pkl.load(open(os.path.join(path, 'epoch.pkl'), 'rb'))
                print('Loading model from', os.path.join(path, 'model', model_path))
                state_dict = torch.load(os.path.join(path, 'model', model_path))
                net.load_state_dict(state_dict)
    else:
        print('Start from scratch!')
        model_path = args.pretrain_model
        state_dict = torch.load(model_path)
        print('turn to the base pretrain model: {}'.format(args.pretrain_model))
        net.load_state_dict(state_dict)
        epoch = 0
        epoch = 0

    return net, epoch


def from_variable_to_numpy(x):
    x = x.data
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.numpy()
    return x


def tile_single(x, action):
    batch_size, c, w, h = x.size()
    assert action.size(0) == batch_size
    action = action.view(action.size(0), -1, 1, 1).repeat(1, 1, w, h)
    return torch.cat([x, action], dim=1)


def tile(x, action):
    return list(map(lambda t: tile_single(t, action), x))


def tile_first(x, action):
    for i in range(len(x) - 1):
        x[i] = tile(x[i], action[:, i, :].float())
    return x

def get_accuracy(output, target):
    tn, fp, fn, tp = confusion_matrix(output, target, labels=[0, 1]).ravel()
    score = (tn + tp) / (tn + fp + fn + tp) * 100.0
    return score

def draw_guide_patch(patch, distribution, guide, radius, line_width=1):
    height, width, _ = patch.shape
    center = (int(width / 2), height)
    for i in range(5):
        r = int(radius / 5 * (5-i))
        cv2.ellipse(patch, center, (r+line_width, r+line_width), 0, -180, 0, (255, 255, 255), -1)
        for j in range(5):
            cv2.ellipse(patch, center, (r, r), 0, -180+36*j, -180+36*j+36, (0, distribution[j*5+4-i]*255, 0), -1)
    for i in range(4):
        angle = math.pi / 5 * (4-i)
        endpoint = (int(center[0] + math.cos(angle) * radius), int(center[1] - math.sin(angle) * radius))
        cv2.line(patch, endpoint, center, (255, 255, 255), line_width)
    return patch

def meshgrid(x, y, row_major=True):
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a+1], 1)
    return torch.cat([a-b/2,a+b/2], 1)

def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    box1, box2 = box1.float(), box2.float()
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    bboxes = bboxes.view(-1, 4)
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        try:
            i = order[0]
        except:
            i = order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def monitor_guide(img, guide, guide_dis, radius=50, line_width=1):
    if guide_dis is not None:
        distribution = guide_dis
    else:
        distribution = np.zeros(25)
    height, width, _ = img.shape
    center = (int(width / 2), height)
    hmin = height-radius-line_width
    wmin = int(width/2-radius-line_width)
    wmax = int(width/2+radius+line_width)
    
    distribution[guide] = 1
    patch = img[hmin:height, wmin:wmax, :].copy()
    patch = draw_guide_patch(patch, distribution, guide, radius, line_width)
    img[hmin:height, wmin:wmax, :] = img[hmin:height, wmin:wmax, :] * 0.5 + patch * 0.5
    return img

def norm_image(images):
    images = images / 255.0
    images = images - 0.5
    images = images * 2.0
    return images
