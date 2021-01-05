from __future__ import division, print_function
from manager import BufferManager
from actionsampler import ActionSampleManager
from utils import generate_guide_grid, color_text, log_seg, get_accuracy, visualize, visualize_guide_action, norm_image
from models import init_models
from retinanet.loss import FocalLoss
import os
import numpy as np
import torch
import torch.nn as nn 
from torch.autograd import Variable
import pickle as pkl
import time
import gc

torch.backends.cudnn.benchmark = True

def update_args(args, new_args):
    if new_args is None:
        return args
    else:
        for k in new_args.keys():
            assert(k in args.keys())
            args[k] = new_args[k]
        return args


def train_colls_with(target, output, acc_func, loss_func, itemname):
    acc = acc_func(target.view(-1).data.cpu().numpy(), 
                torch.max(output.view(-1, 2), -1)[1].data.cpu().numpy())
    print("{0} accuracy: {1:.2f}%".format(itemname, acc))
    item_ls = loss_func()(output.view(-1, 2), target.view(-1).long())
    print('{0} loss: {1:.6f}'.format(itemname, item_ls.data.cpu().numpy()))
    return item_ls


def ins_loss(step, target, output, detect_loss_func, coll_with_loss_func, logger, use_coll_with=False):
    # here we define the loss from instance-level information
    target_cls = target['cls_batch']
    target_loc = target['bboxes_batch']
    target_coll_with = target['colls_with_batch']
    pred_cls = output['cls_pred']
    pred_loc = output['loc_pred']
    pred_coll_with = output['colls_with_prob']

    loss = 0.0
    if not use_coll_with:
        loss = detect_loss_func(pred_loc, target_loc, pred_cls, target_cls)
    else:
        loss = detect_loss_func(pred_loc, target_loc, pred_cls, target_cls, with_coll=True, pred_colls_with=pred_coll_with, target_colls_with=target_coll_with)

    print("bbox loss: {}".format(loss.data.cpu().numpy()))
    return loss


def one_loss(step, target, output, loss_func, field, logger):
    loss = loss_func()(output, target)
    logger.write(step, "{}_loss".format(field), loss.item())
    print("{} loss: {}".format(field, loss.data.cpu().numpy()))
    return loss


def event_losses(step, targets, outputs, acc_func, loss_func, weight_dict, logger):
    # to calculate loss from prediction of future events
    # all these future events are predicted as binary classification
    loss = 0.0
    for itemname in weight_dict.keys():
        event_pred = outputs[itemname + '_prob'].view(-1, 2)
        event_target = targets[itemname + '_batch'].view(-1).long()

        acc = acc_func(event_target.data.cpu().numpy(),
                torch.max(event_pred, -1)[1].data.cpu().numpy())
        print("{0} accuracy: {1:.2f}%".format(itemname, acc))
        
        eloss = one_loss(step, event_target, event_pred, loss_func, itemname, logger)
        weight = weight_dict[itemname]
        loss += eloss * weight
    return loss


def encode_target(target):
    for key in target.keys():
        if key == 'original_bboxes':
            continue
        target[key] = torch.from_numpy(target[key]).float().cuda()
        if key == 'obs_batch':
            # shape: Batch x Pred_step x (3xHistory_len) x H x W
            target[key] = norm_image(target[key])
    return target


class WandBLogger():
    def __init__(self, path, use_logger=True):
        self.use_logger = use_logger
        self.flush_freq = 1000
        self.tmp_str = ""
        self.count = 0
        if self.use_logger:
            self.f = open(path, 'a')
        else:
            pass
    
    def write(self, step, field, value):
        self.count +=1 
        if self.use_logger:
            self.tmp_str += "{} {} {}\n".format(step, field, value)
            if self.count % self.flush_freq == 0:
                self.f.write(self.tmp_str)
                self.tmp_str = ""
                self.f.flush()
            # self.f.write("{} {} {}\n".format(step, field, value))
            # self.f.flush()
        else:
            pass

class Trainer():
    def __init__(self, args, env):
        self.args = args
        assert(self.args.sync)
        self.env = env
        self.max_steps = self.args.max_steps
        self.guides = generate_guide_grid(args.bin_divide)  
        self.bmanager = BufferManager(args) # spc buffer manager
        self.amanager = ActionSampleManager(args, self.guides)   # action sampler
        self.model, self.optim, self.epoch, self.exploration, self.num_steps = init_models(self.args)
        self.env.set_epoch(self.epoch)
        
        # set logger, by default using wandb
        self.logger = WandBLogger(os.path.join(args.save_path, self.args.logger_path), self.args.wandb)

        # import some frequently used params
        self.bsize = self.args.batch_size
        self.pstep = self.args.pred_step
        self.img_h = self.args.frame_height
        self.img_w = self.args.frame_width
        self.classes = self.args.classes    # class number of pixel semantic labels

        # define loss functions used
        self.event_loss_func = nn.CrossEntropyLoss
        self.guide_loss_func = nn.CrossEntropyLoss
        self.speed_loss_func = nn.MSELoss
        self.seg_loss_func = nn.NLLLoss
        self.depth_loss_func = nn.L1Loss()
        self.detect_loss_func = FocalLoss()
        self.coll_with_loss_func = nn.CrossEntropyLoss

        # figure out predictive task list
        self.eventloss_weights = dict() # filed -> loss weight
        self.speedloss_weight = 0.01
        self.segloss_weight = 1.0
        # if self.args.use_detection: self.eventloss_weights['detection'] = 1.0
        # if self.args.use_colls_with: self.eventloss_weights['colls_with'] = 1.0
        if self.args.use_collision: self.eventloss_weights['coll'] = 1.0
        # if self.args.use_collision_other: self.eventloss_weights['coll_other'] = 0.5
        if self.args.use_offroad: self.eventloss_weights['offroad'] = 1.0
        if self.args.use_offlane: self.eventloss_weights['offlane'] = 0.2
        
        self.timer = None
        self.last_episode_step = 0

    def logstream(self, info, reward, total_reward, action, step):
        self.logger.write(step, 'speed', info['speed'])
        self.logger.write(step, 'reward', reward)
        self.logger.write(step, 'episode_reward', total_reward)
        self.logger.write(step, "collision", info["collision"])
        self.logger.write(step, "offroad", info["offroad"])
        self.logger.write(step, "collision_other", info["collision_other"])
        self.logger.write(step, "offlane", info["offlane"])
        
        # print("action [{0:.2f}, {1:.2f}] coll {2} offroad {3} offlane {4} speed {5:.2f} reward {6:.2f} explore {7:.2f}".format(action[0], action[1], info['collision'], info['offroad'],info['offlane'], info['speed'], reward, self.exploration.value(step)))

    def train_model(self, args, step):
        target = self.bmanager.spc_buffer.sample(self.bsize)
        target = encode_target(target)
        target['seg_batch'] = target['seg_batch'].long()

        output = self.model(target['obs_batch'], target['act_batch'], action_var=target['prev_action'])
        loss = 0.0

        batch_thr = self.args.thr
        threshold = batch_thr * self.pstep

        if self.args.use_depth:
            depth_pred = output["depth_pred"].view(-1, self.img_h, self.img_w)
            depth_target = target["depth_batch"].view(-1, self.img_h, self.img_w)
            depth_loss = self.depth_loss_func(depth_pred, depth_target)
            loss += depth_loss
            print("depth loss: {}".format(depth_loss.data.cpu().numpy()))

        if self.args.use_detection:
            original_bboxes = target['original_bboxes']
            bboxes_nums = [[original_bboxes[i][j].size / 5 for j in range(self.pstep + 1)] for i in range(self.bsize)]
            bboxes_ind = [np.array(np.where(np.array(bboxes_nums[i]) > 0)) for i in range(self.bsize)]

            nonempty_batches = []
            empty_batches = []
            for i in range(self.bsize):
                if bboxes_ind[i].size > 0 and 0 in bboxes_ind[i]:
                    # ensure that the first frame in the episode contains at least one vehicle GT
                    nonempty_batches.append(i)
                else:
                    empty_batches.append(i)

            frame_idx = []
            for batch_ind in nonempty_batches:
                for frame_ind in bboxes_ind[batch_ind][0]:
                    frame_idx.append(batch_ind * (self.pstep + 1) + frame_ind)

            if not len(frame_idx) > threshold:
                print(color_text('No enough positive samples to train detector ...', 'green'))
            else:
                # focalloss = FocalLoss()

                '''
                target_cls = target['cls_batch']
                target_loc = target['bboxes_batch']
                target_coll_with = target['coll_with_batch']
                pred_cls = output['cls_pred']
                pred_loc = output['loc_pred']
                pred_coll_with = output['colls_with_prob']
                '''

                # anchor_num = self.bmanager.spc_buffer.anchor_num

                '''
                for bind in nonempty_batches:
                    max_values = [round(pred_cls[bind, i].sigmoid().max().cpu().item(), 2) for i in range(pred_cls.size(1))]
                    print("{}: {}".format(bind, max_values))
                for bind in empty_batches:
                    max_values = [round(pred_cls[bind, i].sigmoid().max().cpu().item(), 2) for i in range(pred_cls.size(1))]
                    print("{}: {}".format(bind, max_values))
                '''

                # Strategy #2: put frames with at least one vehicle instance into training
                '''
                pred_loc = pred_loc.view(-1, anchor_num, 4)[frame_idx]
                target_loc = target_loc[:, :, :, :4].view(-1, anchor_num, 4)[frame_idx]
                pred_cls = pred_cls.view(-1, anchor_num, 1)[frame_idx]
                target_cls = target_cls.view(-1, anchor_num)[frame_idx]
                pred_coll_with = pred_coll_with.view(-1, anchor_num, 2)[frame_idx]
                target_coll_with = target_coll_with.view(-1, anchor_num)[frame_idx]
                '''

                # detectloss = FocalLoss()
                instance_loss = ins_loss(step, target, output, self.detect_loss_func, self.coll_with_loss_func, self.logger, use_coll_with=self.args.use_colls_with)

                '''
                bbox_ls = detect_loss(pred_loc, target_loc, pred_cls, target_cls)
                loss += bbox_ls  
                self.logger.write(step, "detect_loss", bbox_ls.item())

                if args.use_colls_with:
                    pred_colls_with = output['colls_with_prob']
                    pred_colls_with = torch.cat([pred_colls_with[idx, bboxes_ind[idx][0], :, :] for idx in nonempty_batches], axis=0).view(-1, anchor_num)
                    target_colls_with = target['colls_with_batch']
                    target_colls_with = torch.cat([target_colls_with[idx, bboxes_ind[idx][0], :, :] for idx in nonempty_batches], axis=0).view(-1, anchor_num)
                    colls_with_loss = train_colls_with(target_colls_with, pred_colls_with, get_accuracy, self.event_loss_func, "colls_with")
                    loss += colls_with_loss
                '''
                loss += instance_loss

        # Loss Part #2: loss from future event happening prediction
        loss += event_losses(step, target, output, get_accuracy, self.event_loss_func, self.eventloss_weights, self.logger)

        # Loss Part #3: loss from future speed prediction
        if args.use_speed:
            speed_pred = output['speed']
            speed_target = target['sp_batch'][:, 1:].unsqueeze(dim=2)
            speedloss = one_loss(step, speed_target, speed_pred, self.speed_loss_func, "speed", self.logger)
            loss += self.speedloss_weight * speedloss

        # Loss Part #3: loss from future pixelwise semantic label prediction
        seg_pred = output['seg_pred'].view(-1, self.classes, self.img_h, self.img_w)
        seg_target = target['seg_batch'].view(-1, self.img_h, self.img_w)
        segloss = one_loss(step, seg_target, seg_pred, self.seg_loss_func, "seg", self.logger)
        loss += self.segloss_weight * segloss

        self.logger.write(step, "total_loss", loss.item())
        gc.collect()
        return loss

    def save(self, step):
        print(color_text('Saving models ...', 'green'))
        torch.save(self.model.module.state_dict(),
                        os.path.join(self.args.save_path, 'model', 'pred_model_%09d.pt' % step))
        torch.save(self.optim.state_dict(),
                    os.path.join(self.args.save_path, 'optimizer', 'optimizer.pt'))
        with open(os.path.join(self.args.save_path, 'epoch.pkl'), 'wb') as f:
            pkl.dump(self.epoch, f)
        self.bmanager.save_spc_buffer()
        print(color_text('Model saved successfully!', 'green'))

    def train_guide_action(self, step):
        if self.bmanager.spc_buffer.can_sample_guide(self.bsize):
            obs, guide_action = self.bmanager.spc_buffer.sample_guide(self.bsize)
            q = self.model(obs, action_only=True)
            loss = self.guide_loss_func()(q, guide_action)
            print('Guidance loss  %0.4f' % loss.data.cpu().numpy())
            return loss
        else:
            print(color_text('Insufficient expert data for imitation learning.', 'red'))
            return 0.0

    def train_spn(self, step):
        # to train the semantic predictive network
        self.model.train()
        for ep in range(self.args.num_train_steps):
            self.optim.zero_grad()
            pred_loss = self.train_model(self.args, ep+step)
            guide_loss = self.train_guide_action(ep+step)
            loss = pred_loss + guide_loss
            try:
                print('loss = %0.4f\n' % loss.data.cpu().numpy())
            except:
                print('loss = %0.4f\n' % loss)
            loss.backward()
            self.optim.step()
            self.epoch += 1

        if self.epoch % self.args.save_freq == 0:
            self.save(step)


    def summarize(self, num_episode, total_reward, step):
        # summarize after each episode ends
        end_time = time.time()
        episode_time = end_time - self.timer
        self.timer = time.time()

        episode_step = step - self.last_episode_step
        self.last_episode_step = step 

        print("-------- episode {} ---------".format(num_episode))
        print("reward: {}".format(total_reward))
        print("steps: {}".format(episode_step))
        print("time: {} | {}/step".format(episode_time, episode_time/episode_step))


    def run(self, extra_args=None):
        self.args = update_args(self.args, extra_args)
        action_var = Variable(torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, self.args.frame_history_len - 1, 1), requires_grad=False).float()

        obs, info = self.env.reset()
        num_episode = 1
        total_reward = 0 
        last_episode = 1

        self.timer = time.time()
        self.last_episode_step = 0

        print("Start training ...")

        for step in range(self.num_steps, self.max_steps):
            obs_var = self.bmanager.store_frame(obs, info)
            self.model.eval()
            action, guide_action = self.amanager.sample_action(net=self.model, obs=obs, obs_var=obs_var,action_var=action_var, exploration=self.exploration, step=step, explore=num_episode % 2)

            obs, reward, done, info = self.env.step(action)
            action_var = self.bmanager.store_effect(guide_action, action, reward, done, info)
            
            total_reward += reward
            self.logstream(info, reward, total_reward, action, step)

            if self.bmanager.spc_buffer.can_sample(self.bsize) \
                and self.args.sync and step % self.args.learning_freq == 0:
                # Note, here only sync mode is supported, so it cannot be used on Torcs any more
                self.train_spn(step)

            if done:
                self.summarize(num_episode, total_reward, step)
                num_episode += 1
                total_reward = 0
                obs, info = self.env.reset()
                self.bmanager.reset(step)
                self.amanager.reset()
                gc.collect()


def train_policy(args, env):
    trainer = Trainer(args, env)
    trainer.run()
