from __future__ import division, print_function
import numpy as np
import os
import random
import torch
from torch.autograd import Variable
from utils.dataset import DataEncoder
import gc
import json
from utils import norm_image


class SPCBuffer(object):
    def __init__(self, args):
        self.args = args
        self.next_idx = 0
        self.num_in_buffer = 0
        self.last_idx = 0
        self.obs = None
        self.action = None
        self.done = None
        self.collision = None
        self.collision_other = None
        self.collision_vehicles = None
        self.colls_with = None
        self.offroad = None
        self.offlane = None
        self.speed = None
        self.seg = None
        self.bboxes = None
        self.depth = None
        self.directions = None
        self.bboxes_cls = None
        self.expert = None
        self.guide_action = None
        self.epi_lens = []
        self.bbox_encoder = DataEncoder()
        width, height = self.args.frame_width, self.args.frame_height
        anchors = self.bbox_encoder._get_anchor_boxes(input_size=torch.Tensor((width, height)))
        self.anchor_num = anchors.size(0)

    def can_sample_guide(self, batch_size):
        # determines whether there are enough expert data for self-imitation learning
        if len(self.epi_lens) == 0:
            return False
        bar = self.get_bar()
        bar_index = np.where(self.expert[:self.num_in_buffer] >= bar)[0]
        if self.args.verbose:
            print('Calculating bar from %s' % str(self.epi_lens))
            print('Bar: %d' % bar)
            print('Number of candidates: %d' % len(bar_index))
        return len(bar_index) >= batch_size

    def get_bar(self):
        # calculate the bar according to which expert guidance data are selected
        idx = int(len(self.epi_lens) * self.args.expert_ratio)
        bar = max(sorted(self.epi_lens, reverse=True)[idx], self.args.expert_bar)
        return bar

    def sample_guide(self, batch_size):
        # sample expert guidance replay data for self-imitation learning
        indices = np.where(self.expert[:self.num_in_buffer] >= self.get_bar())[0]
        indices = list(np.random.choice(list(indices), batch_size))
        obs = torch.from_numpy(np.concatenate([self.obs[idx][np.newaxis, :] for idx in indices], axis=0)).float()
        obs = norm_image(obs)
        guide_action = Variable(torch.from_numpy(self.guide_action[indices]), requires_grad=False).long()
        if torch.cuda.is_available():
            obs = obs.cuda()
            guide_action = guide_action.cuda()
        return obs, guide_action

    def sample_n_unique(self, sampling_f, n):
        res = []
        while len(res) < n:
            candidate = sampling_f()
            done = self.sample_done(candidate)
            if candidate not in res and done:
                res.append(candidate)  
        return res

    def sample_done(self, idx):
        if idx < 10 or idx >= self.num_in_buffer - self.args.pred_step - 10:
            return False
        else:
            done_list = self.done[idx - self.args.frame_history_len + 1: idx + self.args.pred_step + 1]
            if np.sum(done_list) >= 1.0:
                return False
            else:
                return True

    def can_sample(self, batch_size):
        return (batch_size * (self.args.pred_step + 1) + 20 + self.args.pred_step <= self.num_in_buffer)

    def update_epi(self, idx_buffer, safe_buffer, epi_len):
        self.expert[idx_buffer] = safe_buffer
        self.epi_lens.append(epi_len)

    def _encode_sample(self, indices):
        data_dict = dict()

        data_dict['obs_batch'] = np.concatenate([np.concatenate([self._encode_observation(idx + ii)[np.newaxis, :] for ii in range(1)], 0)[np.newaxis, :] for idx in indices], axis=0)
        data_dict['act_batch'] = np.concatenate([self.action[idx: idx+self.args.pred_step, :][np.newaxis, :] for idx in indices], axis=0)
        data_dict['sp_batch'] = np.concatenate([self.speed[idx: idx+self.args.pred_step+1][np.newaxis, :] for idx in indices], axis=0)
        data_dict['prev_action'] = np.concatenate([self.action[idx-self.args.frame_history_len + 1: idx, :][np.newaxis, :] for idx in indices], axis=0)
        data_dict['seg_batch'] = np.concatenate([self.seg[idx: idx+self.args.pred_step+1, :][np.newaxis, :] for idx in indices], axis=0)

        if self.args.use_collision:
            data_dict['coll_batch'] = np.concatenate([self.collision[idx+1: idx + self.args.pred_step + 1][np.newaxis, :] for idx in indices], axis=0)
            data_dict['coll_other_batch'] = np.concatenate([self.collision_other[idx+1: idx + self.args.pred_step + 1][np.newaxis, :] for idx in indices], axis=0)
            data_dict['coll_vehicles_batch'] = np.concatenate([self.collision_vehicles[idx+1: idx + self.args.pred_step + 1][np.newaxis, :] for idx in indices], axis=0)
        if self.args.use_offroad:
            data_dict['offroad_batch'] = np.concatenate([self.offroad[idx+1: idx + self.args.pred_step + 1][np.newaxis, :] for idx in indices], axis=0)
        if self.args.use_offlane:
            data_dict['offlane_batch'] = np.concatenate([self.offlane[idx+1: idx + self.args.pred_step + 1][np.newaxis, :] for idx in indices], axis=0)

        if self.args.use_depth:
            data_dict["depth_batch"] = np.concatenate([self.depth[idx: idx+self.args.pred_step+1, :][np.newaxis, :] for idx in indices], axis=0)

        if self.args.use_detection:
            bboxes_batch = np.zeros([len(indices), self.args.pred_step+1, self.anchor_num, 4], dtype=np.float16)
            cls_batch = np.zeros([len(indices), self.args.pred_step+1, self.anchor_num], dtype=np.int8)
            colls_with_batch = np.zeros([len(indices), self.args.pred_step+1, self.anchor_num], dtype=np.int8)
            # coll_with_batch = np.zeros([len(indices), self.args.pred_step+1, self.anchor_num], dtype=np.int8)
            original_bboxes_batch = []
            for i in range(len(indices)):
                original_bboxes = []
                idx = indices[i]
                for j in range(self.args.pred_step + 1):
                    bboxes = np.array(self.bboxes[idx + j])
                    labels = np.array(self.bboxes_cls[idx + j])
                    colls_with = np.array(self.colls_with[idx + j])
                    bboxes_orientations = np.array([self.bboxes[idx + j][u] for u in range(len(self.bboxes[idx + j]))])
                    original_bboxes.append(bboxes_orientations)
                    if bboxes.shape[0] == 0:
                        bboxes_batch[i, j, :, :4] = 0
                        cls_batch[i, j, :] = -1
                        colls_with_batch[i, j, :] = -1
                    else:
                        bboxes = torch.Tensor(bboxes)
                        labels = torch.Tensor(labels)
                        bboxes_batch[i, j, :, :4], cls_batch[i, j, :], colls_with_batch[i, j, :] = self.bbox_encoder.encode(bboxes, labels, colls_with, input_size=(self.args.frame_width, self.args.frame_height))
                original_bboxes_batch.append(original_bboxes)
            data_dict['bboxes_batch'] = bboxes_batch
            data_dict['cls_batch'] = cls_batch
            data_dict['colls_with_batch'] = colls_with_batch
            data_dict['original_bboxes'] = original_bboxes_batch
    
        return data_dict

    def decode_bbox(self, loc_preds, cls_preds, batchsize):
        return self.bbox_encoder.decode(loc_preds, cls_preds, input_size=(self.args.frame_width, self.args.frame_height), batchsize=batchsize)

    def decode_one(self, loc_preds, cls_preds, inputsize):
        return self.bbox_encoder.decode_one(loc_preds, cls_preds, inputsize)

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        indices = self.sample_n_unique(lambda: random.randint(10, self.num_in_buffer - 10), batch_size) 
        return self._encode_sample(indices)

    def _encode_observation(self, idx):
        start_idx = idx - self.args.frame_history_len + 1
        end_idx = idx + 1
        assert start_idx >= 0 and end_idx <= min(self.num_in_buffer, self.args.buffer_size) and np.sum(self.done[start_idx: end_idx]) == 0
        encoded_obs = self.obs[start_idx: end_idx].reshape(-1, self.args.frame_height, self.args.frame_width)
        return encoded_obs

    def store_frame(self, obs, collision, collision_other, collision_vehicles, coll_with, offroad, offlane, speed, seg, bboxes, depth):
        # as the convention in opencv, we operate and store image in CxHxW format        
        frame = obs.transpose(2, 0, 1)  # reshape as [C, H, W]

        if self.obs is None:
            self.obs = np.empty([self.args.buffer_size, 3, self.args.frame_height, self.args.frame_width], dtype=np.uint8)
            self.action = np.empty([self.args.buffer_size, self.args.num_total_act], dtype=np.float16)
            self.done = np.empty([self.args.buffer_size], dtype=np.int8)
            self.expert = np.empty([self.args.buffer_size], dtype=np.float16)
            self.guide_action = np.empty([self.args.buffer_size], dtype=np.int8)
            self.collision = np.empty([self.args.buffer_size], dtype=np.int8)
            self.collision_other = np.empty([self.args.buffer_size], dtype=np.int8)
            self.collision_vehicles = np.empty([self.args.buffer_size], dtype=np.int8)
            self.offroad = np.empty([   self.args.buffer_size], dtype=np.int8)
            self.offlane = np.empty([self.args.buffer_size], dtype=np.int8)
            self.speed = np.empty([self.args.buffer_size], dtype=np.float16)
            self.seg = np.empty([self.args.buffer_size, self.args.frame_height, self.args.frame_width], dtype=np.uint8)
            self.depth = np.empty([self.args.buffer_size, self.args.frame_height, self.args.frame_width], dtype=np.float16)

            # because the ground truth bboxes number varies in different frames, we can't allocate a numpyarray to hold them
            self.bboxes = [[] for i in range(self.args.buffer_size)]
            # self.directions = [[] for i in range(self.args.buffer_size)]
            self.bboxes_cls = [[] for i in range(self.args.buffer_size)]
            self.colls_with = [[] for i in range(self.args.buffer_size)]

        self.obs[self.next_idx] = frame
        self.collision[self.next_idx] = int(collision)
        self.collision_other[self.next_idx] = int(collision_other)
        self.collision_vehicles[self.next_idx] = int(collision_vehicles)
        self.offroad[self.next_idx] = int(offroad)
        self.offlane[self.next_idx] = int(offlane)
        self.speed[self.next_idx] = speed
        self.seg[self.next_idx, :] = seg
        self.depth[self.next_idx, :] = depth

        if self.args.use_detection:
            labels = [0 for i in range(len(bboxes))] # curently we only detect the vehicles
            self.bboxes[self.next_idx] = bboxes
            # self.directions[self.next_idx] = directions
            self.bboxes_cls[self.next_idx] = labels
            self.colls_with[self.next_idx] = list(coll_with)

        self.last_idx = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.args.buffer_size
        
        self.num_in_buffer = min(self.args.buffer_size, self.num_in_buffer + 1)

        gc.collect()

    def store_action(self, guide_action, action, done):
        self.guide_action[self.last_idx] = guide_action
        self.action[self.last_idx, :] = action
        self.done[self.last_idx] = int(done)

    '''
    # this function is replaced by the two buffer classes in manager.py
    def get_history(self, target):
        if target == 'action':
            target_buffer = self.action
            his_len = self.args.frame_history_len - 1
        elif target == 'obs':
            target_buffer = self.obs
            his_len = self.args.frame_history_len
        else:
            assert(0)
        
        if self.num_in_buffer < his_len:
            # no enough history stored in the buffer
            history_seq = [target_buffer[self.last_idx] for i in range(his_len)]
        else:
            history_seq = []
            for i in range(his_len):
                idx = self.last_idx - (his_len - 1) + i
                idx = self.num_in_buffer + idx if idx < 0 else idx
                history_seq.append(target_buffer[idx])
        
        return np.concatenate(history_seq, 0)[np.newaxis, ]
    '''

    def load(self, path):
        if self.args.eval:
            print('not load spc buffers in eval mode...')
            return
        spc_path = os.path.join(self.args.save_path, 'spc_checkpoint')
        if os.path.exists(spc_path):
            print('load the spcbuffer checkpoint ...')
            file_list = os.listdir(spc_path)
            for filename in file_list:
                if filename[-4:] == '.npy':
                    name = filename[:-4]
                    filepath = os.path.join(spc_path, filename)
                    self.__dict__[name] = np.load(filepath)
                if filename == 'others.json':
                    filepath = os.path.join(spc_path, filename)
                    var_dict = json.load(open(filepath, 'r'))
                    for key in var_dict:
                        self.__dict__[key] = var_dict[key]
            print("successfully load the spcbuffer checkpoint")

    def save(self, path):
        # In case the whole class is too large to save, we independently save different components
        spc_path = os.path.join(self.args.save_path, 'spc_checkpoint')
        if not os.path.isdir(spc_path):
            os.makedirs(spc_path)
        save_dict = {} 
        for key in self.__dict__.keys():
            component = self.__dict__[key]
            if type(component) == np.ndarray:
                np.save(os.path.join(spc_path, '{}.npy'.format(key)), component)
            elif key == "bboxes":
                bboxes = np.array(component)
                np.save(os.path.join(spc_path, '{}.npy'.format(key)), component)
            elif type(component) == int or type(component) == list:
                # import pdb; pdb.set_trace()
                save_dict[key] = component
        with open(os.path.join(spc_path, 'others.json'), 'w') as f:
            try:
                json.dump(save_dict, f)
            except:
                import pdb; pdb.set_trace()
    
