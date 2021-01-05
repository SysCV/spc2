import os
import copy
import numpy as np
import torch
from utils import setup_logger
from torch.autograd import Variable
from spcbuffer import SPCBuffer
import time


class BufferManager:
    class ObsBuffer:
        def __init__(self, frame_history_len=3):
            self.frame_history_len = frame_history_len
            self.last_obs_all = []

        def store_frame(self, frame):
            obs_np = frame.transpose(2, 0, 1)
            if not len(self.last_obs_all) == self.frame_history_len:
                self.last_obs_all = [obs_np for i in range(self.frame_history_len)]
            else:
                self.last_obs_all = self.last_obs_all[1:] + [obs_np]
            return np.concatenate(self.last_obs_all, 0)

        def clear(self):
            self.last_obs_all = []
            return

    class ActionBuffer:
        def __init__(self, frame_history_len=3):
            self.frame_history_len = frame_history_len
            self.last_action_all = []

        def store_frame(self, action):
            action = action.reshape(1, -1)
            if not len(self.last_action_all) == self.frame_history_len:
                self.last_action_all = [action for i in range(self.frame_history_len)]
            else:
                self.last_action_all = self.last_action_all[1:] + [action]
            return np.concatenate(self.last_action_all, 0)[np.newaxis, ]

        def clear(self):
            self.last_action_all = []
            return

    def __init__(self, args=None):
        self.args = args
        mode = 'eval' if args.eval else 'train'
        self.reward_logger = setup_logger(mode, os.path.join(args.save_path, 'reward_{}_{}.txt'.format(mode, args.env)), resume=self.args.resume)

        self.spc_buffer = SPCBuffer(args)
        if args.resume:
            self.spc_buffer.load(args.save_path)
        self.obs_buffer = self.ObsBuffer(args.frame_history_len)
        self.action_buffer = self.ActionBuffer(args.frame_history_len - 1)

        self.prev_act = np.array([1.0, 0.0])
        self.reward = 0.0
        self.collision_buffer = []
        self.offroad_buffer = []
        self.offlane_buffer = []
        self.idx_buffer = []

        self.dist_sum = 0.0

    def store_frame(self, obs, info):  
        past_n_frames = self.obs_buffer.store_frame(obs)

        obs_var = Variable(torch.from_numpy(past_n_frames).unsqueeze(0).float().cuda())

        self.spc_buffer.store_frame(obs=obs,
                                    collision=info['collision'],
                                    collision_other=info['collision_other'],
                                    collision_vehicles=info['collision_vehicles'],
                                    coll_with=info['coll_with'],
                                    offroad=info['offroad'],
                                    offlane=info['offlane'],
                                    speed=info['speed'],
                                    seg=info['seg'],
                                    bboxes=info["bboxes"],
                                    depth=info['depth'])
        self.idx_buffer.append(self.spc_buffer.last_idx)
        self.dist_sum += info['speed']

        return obs_var

    def store_effect(self, guide_action, action, reward, done, info):
        self.collision_buffer.append(info['collision'])
        self.offroad_buffer.append(info['offroad'])
        self.offlane_buffer.append(info['offlane'])
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False).float()
        self.spc_buffer.store_action(guide_action, action, done)
        
        self.reward += reward
        return act_var

    def reset(self, step):
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.prev_act = np.array([1.0, 0.0])

        self.reward_logger.info('step {} reward {}'.format(step, self.reward))

        # construct labels for self-imitation learning
        epi_len = len(self.idx_buffer)
        idx_buffer = np.array(self.idx_buffer)
        collision_buffer = np.array(self.collision_buffer)
        collision_buffer = np.array([np.sum(collision_buffer[i:i + self.args.safe_length_collision]) == 0 for i in range(collision_buffer.shape[0])])
        offroad_buffer = np.array(self.offroad_buffer)
        offroad_buffer = np.array([np.sum(offroad_buffer[i:i + self.args.safe_length_offroad]) == 0 for i in range(offroad_buffer.shape[0])])
        offlane_buffer = np.array(self.offlane_buffer)
        offlane_buffer = np.array([np.sum(offlane_buffer[i:i + self.args.safe_length_offlane]) == 0 for i in range(offlane_buffer.shape[0])])
        safe_buffer = collision_buffer * offroad_buffer * offlane_buffer * self.dist_sum 
        self.spc_buffer.update_epi(idx_buffer, safe_buffer, epi_len)

        self.idx_buffer = []
        self.collision_buffer = []
        self.offroad_buffer = []
        self.offlane_buffer = []
        self.dist_sum = 0.0
        self.reward = 0.0

    def save_spc_buffer(self):
        # Saving an object larger than 4 GiB causes overflow error
        self.spc_buffer.save(self.args.save_path)

    def load_spc_buffer(self):
        self.spc_buffer.load(self.args.save_path)
