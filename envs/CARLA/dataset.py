# define the dataset used in training on offline CARLA dataset
import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import math
from utils import meshgrid, box_iou, box_nms, change_box_order
import random


class AngleSpliter:
    # class to spliter the original angle into bin and residual
    def __init__(self, bins=4, overlap=0.05):
        self.bins = bins
        # in the original 3d-deep-box design, bins have overlap with
        # neighboring bins, but here, for simplicity, we eliminate that
        self.overlap = overlap
        self.angle_bins = np.zeros(self.bins)

        self.interval = 2 * np.pi / self.bins + self.overlap
        for i in range(1, self.bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2.0 # move to the center of the bin

        self.bins_ranges = [] 
        for i in range(0, bins):
            lower = (i*self.interval) 
            upper = (i*self.interval + self.interval) 
            self.bins_ranges.append((lower, upper))

    def get_bin(self, angle):
        bin_idxs = []
        sins, coss = [], []
        def is_between(min, max, angle):
            if min < max:
                # usual
                return (min <= angle) and (angle < max)
            else:
                return (min > angle) or (angle > max)

        for bin_idx, bin_range in enumerate(self.bins_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)
                residual = angle - bin_range[0]
                sins.append(math.sin(residual))
                coss.append(math.cos(residual))
                # residuals.append(angle - bin_range[0])

        return bin_idxs, sins, coss

    def get_bin_group(self, angle_list):
        bins = []
        sinses, cosses = [], []
 
        for angle in angle_list:
            bin_idx, sins, coss = self.get_bin(angle)
            bins.append(bin_idx)
            sinses.append(sins)
            cosses.append(coss)


        if [] in bins:
            print(angle_list, bins, sinses, cosses)

        return torch.Tensor(bins), torch.Tensor(sinses), torch.Tensor(cosses)

    def split(self, angle):
        # the input angle is of 360 degree format
        angle = angle % (2*np.pi)
        bin_idxs, sines, coses = self.get_bin_group(angle)
        angle_num = angle.shape[0]

        # bin_idxs = self.get_bin(angle)
        # arget_bins = torch.zeros([angle_num, self.bins])
        # for aind in range(angle_num):
        #     target_bins[aind][bin_idxs[aind]] = 1
        '''
        for i in range(self.bins):
            (lower, upper) = self.bins_ranges[i]
            residual = angle - lower
            residuals[:, i] = residual
        '''

        return bin_idxs, sines, coses


class FullStackEncoder:
    # encode all instance-level information into anchor-based representations
    def __init__(self):
        # self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.anchor_areas = [16*16., 32*32., 64*64., 128*128., 256*256.] # p2 -> p6
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args: 
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)] # p2 -> p6
        # fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, center_points, labels, colls_with, dimensions, bins, sines, coses, input_size):
        bins = bins.squeeze(1)
        sines = sines.squeeze(1)
        coses = coses.squeeze(1)

        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          colls_with: (tensor) whether the vehicle collides with the player agent, sized [#obj] (binary)
          dimensions: (tensor), sized [#obj, 3]
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        
        try:
            boxes = change_box_order(boxes, 'xyxy2xywh')
        except:
            assert(0)
            print("a vehicle-free frame, which should be eliminated in a clean dataset")
            boxes = torch.Tensor([[0., 0., 0., 0.]])
            colls_with = torch.Tensor([0.])
            dimensions = torch.Tensor([[0., 0., 0.]])
            sines = torch.Tensor([0.])
            coses = torch.Tensor([0.])
            bins = torch.Tensor([0.])
            labels = torch.Tensor([0.])
            # orientations = torch.Tensor([0.])

        colls_with = torch.Tensor(colls_with)
        colls_with = colls_with
        dimensions = dimensions
        # orientations = orientations.float()

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)

        # select matching instance
        boxes = boxes[max_ids]
        center_points = center_points[max_ids]
        colls_with = colls_with[max_ids]
        dimensions = dimensions[max_ids]
        cls_targets = labels[max_ids]
        bins = bins[max_ids]
        sines = sines[max_ids]
        coses = coses[max_ids]
        # orientations = orientations[max_ids]

        # build offset referring to target anchors
        # print(boxes[0,0], "before more")
        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        # print(loc_xy[0, 0], loc_wh[0,0], "before")
        loc_targets = torch.cat([loc_xy,loc_wh], 1) # sized [num_anchor, 4]
        center_xy = (center_points[:, :2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        center_depth = center_points[:, 2].unsqueeze(1)
        center_targets = torch.cat([center_xy, center_depth], 1) # sized [num_anchor, 3]

        # filter invalid or negative instance
        sines[max_ious<0.5] = 0
        coses[max_ious<0.5] = 0
        bins[max_ious<0.5] = 0
        cls_targets[max_ious<0.5] = 0
        colls_with[max_ious<0.5] = 0
        dimensions[max_ious<0.5] = 0
        # orientations[max_ious<0.5] = 0

        # ignore some not enough overlapped instances
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        colls_with[ignore] = -1
        dimensions[ignore] = -1
        bins[ignore] = -1
        
        # colls_with[ignore] = -1
        # print(loc_targets[0, 0], "in encoder")
        return loc_targets, cls_targets, center_targets, colls_with, dimensions, bins, sines, coses

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = 0.5
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]
        
        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()             # [#obj,]
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)

        try:
            res_boxes = boxes[ids][keep]
            res_labels = labels[ids][keep]
            res_scores = score[ids][keep]
        except:
            res_boxes = boxes[ids].unsqueeze(0)
            res_labels = labels[ids].unsqueeze(0)
            res_scores = score[ids].unsqueeze(0)
        return res_boxes, res_labels, res_scores

        # return boxes[ids][keep], labels[ids][keep]


class Episode_Handler(object):
        # class to hold valid episode in the dataset
        def __init__(self, path, episode, start_frame, end_frame, transform, fetch_len=5):
            self.path = path 
            self.episode = str(episode)
            self.start_frame = start_frame
            self.fetch_len = fetch_len
            self.end_frame = end_frame
            self.fnum = end_frame - start_frame + 1
            assert fetch_len <= self.fnum, "to fetch a sequence longer than episode length"
            self.max_start_index = self.fnum - fetch_len
            self.data_pool = self.build_data_pool()
            self.transform = transform
            self.encoder = FullStackEncoder()
            self.anglespliter = AngleSpliter()

        def build_data_pool(self):
            data_pool = dict()
            subkeys = ["2d_bbox", "3d_bbox", "depth", "dimensions", "orientations", "seg"]
            for key in subkeys:
                data_pool[key] = []
                for findex in range(self.start_frame, self.end_frame+1):
                    fpath = os.path.join(self.path, self.episode, key, "{}.npy".format(findex))
                    data_pool[key].append(fpath)
                
            data_pool["obs"] = []
            data_pool["extrinsic"] = []
            data_pool["intrinsic"] = []
            data_pool["camera_transform"] = []
            data_pool["player_transform"] = []
            data_pool["action"] = []
            data_pool["ins_level"] = []
            data_pool["scene_level"] = []

            action_f = os.path.join(self.path, self.episode, "action.txt")
            action_lines = open(action_f).readlines()
            state_f = os.path.join(self.path, self.episode, "state.txt")
            state_lines = open(state_f).readlines()
            coll_with_f = os.path.join(self.path, self.episode, "coll_withs.txt")
            coll_with_lines = open(coll_with_f).readlines()
            
            for findex in range(self.start_frame, self.end_frame+1):
                obs_path = os.path.join(self.path, self.episode, "obs", "{}.jpg".format(findex))
                data_pool["obs"].append(obs_path)
                for k in ["extrinsic", "intrinsic", "camera_transform", "player_transform"]:
                    calib_path = os.path.join(self.path, self.episode, "calib", "{}_{}.npy".format(k, findex))
                    data_pool[k].append(calib_path)

                action_line = action_lines[findex].strip()
                state_line = state_lines[findex].strip()
                coll_with_line = coll_with_lines[findex].strip()
                items = action_line.split()
                steer, throttle = float(items[1]), float(items[2])
                data_pool["action"].append((steer, throttle))
                colls_with = coll_with_line.split("[")[1].split("]")[0].split()
                colls_with = [float(x) for x in colls_with]
                data_pool["ins_level"].append(colls_with)
                state = state_line.split()[1:]
                state = [int(x) for x in state]
                data_pool["scene_level"].append(torch.Tensor(state))

            return data_pool
                
        def fetch(self):  
            # to save memory, encoding is performed until the data is fetched
            start_index = random.randint(0, self.max_start_index)
            output = dict()
            for k in ["obs", "seg", "depth", "loc_targets", "cls_targets", "center_targets", "dimensions", "sines", "coses", "bins", "action", "extrinsic", "intrinsic", "state", "ins_state"]:
                output[k] = []

            for index in range(start_index, start_index+self.fetch_len):
                obs = self.transform(Image.open(self.data_pool["obs"][index]))

                tmp_dict = dict()
                for k in ["seg", "2d_bbox", "3d_bbox", "depth", "dimensions", "orientations", "extrinsic", "intrinsic"]:
                    tmp_dict[k] = torch.from_numpy(np.load(self.data_pool[k][index]).astype(np.float32))

                action = torch.Tensor(self.data_pool["action"][index])
                state = torch.Tensor(self.data_pool["scene_level"][index])
                ins_state = self.data_pool["ins_level"][index]

                # preprocessing and encoding information by anchors
                tmp_dict["orientations"] = tmp_dict["orientations"] / 360.0 * 2 * np.pi
                bins, sines, coses = self.anglespliter.split(tmp_dict["orientations"])
                box_num = tmp_dict["2d_bbox"].shape[0]
                labels = torch.ones(box_num)
                center_points = torch.mean(tmp_dict["3d_bbox"], axis=1)
                ins_state = [0.0 for i in range(box_num)] # to be fixed!
                loc_targets, cls_targets, center_targets, ins_state, dimension, bins, sines, coses = self.encoder.encode(tmp_dict["2d_bbox"], center_points, labels, ins_state, tmp_dict["dimensions"], bins, sines, coses, (512, 256))

                output["obs"].append(obs)
                output["seg"].append(tmp_dict["seg"])
                output["depth"].append(tmp_dict["depth"])
                output["loc_targets"].append(loc_targets)
                output["cls_targets"].append(cls_targets)
                output["center_targets"].append(center_targets)
                output["dimensions"].append(dimension)
                output["sines"].append(sines)
                output["coses"].append(coses)
                output["bins"].append(bins)
                output["action"].append(action)
                output["state"].append(state)
                output["ins_state"].append(ins_state)
                output["extrinsic"].append(tmp_dict["extrinsic"])
                output["intrinsic"].append(tmp_dict["intrinsic"])
                
            return output


class CarlaDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size,
        udepth=True, u2d=True, u3d=True, useg=True, history_len=3, pred_step=5):
        '''
        dataset implementation with both 2d and 3d detection infomation
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size (W,H).
          udepth, u2d, u3d, useg: whether using depth, 2d/3d-detection or semantic-seg 
          history_len: how long the historical frame-sequence is used
          pred_step: to predict how many steps into the future
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.use_depth, self.use_2d_detection, self.use_3d_detection, self.use_seg = udepth, u2d, u3d, useg
        self.his_len = history_len
        self.pred_step = pred_step
        self.actions = dict()
        self.states = dict()
        self.encoder = FullStackEncoder()
        self.anglespliter = AngleSpliter()

        '''
        NOTE:
            the difficulty to build the dataset is that, instead of retrive a singal frame, it
            has to return a sequence of frames with both previous and future frames in the following
            getitem function, with a given index. So here, we need to know how many steps can be
            regarded as the "current frame" in data retrival
        NOTE: 
            the number of total frames is not simply the frames collected in the dataset but the number of 
            legal "current frame" to start with. For example, if an episode has N frames, history_len=3, 
            pred_step=5, the the legal frames to start with in this episode is: N-3-5+1 = N-7
        '''
        self.num_frame = 0
        self.content = [] # containing (episode, step) pairs
        self.episodes = []

        with open(list_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # "{episode}: start_frame - end_frame"
                items = line.split()
                episode = int(items[0][:-1])
                start_frame = int(items[1])
                end_frame = int(items[3])
                episode_handler = Episode_Handler(self.root, episode, start_frame, end_frame, self.transform, self.his_len + self.pred_step)
                self.episodes.append(episode_handler)

        self.episode_num = len(self.episodes) 


    def __getitem__(self, idx):
        output = self.episodes[idx].fetch()
        for k in output.keys():
            output[k] = torch.stack(output[k])
        return output


    def collate_fn(self, package):
        batch = dict()
        for k in package[0].keys():
            batch[k] = torch.stack([seq[k] for seq in package])
        return batch

    def __len__(self):
        # return self.num_frame
        return self.episode_num


def build_data_loader(root, args, train_list, test_list, transform, size):
    # build train and test data loaders in given path
    trainset = CarlaDataset(root=root, list_file=train_list, train=True, transform=transform, input_size=size)
    testset = CarlaDataset(root=root, list_file=test_list, train=False, transform=transform, input_size=size)
    trainloader = torch.utils.data.DataLoader(trainset, args.batchsize, shuffle=True, num_workers=1, collate_fn=trainset.collate_fn)
    testloader = torch.utils.data.DataLoader(testset, args.batchsize, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)
    return trainloader, testloader