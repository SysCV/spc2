from __future__ import division, print_function
from manager import BufferManager
from actionsampler import ActionSampleManager
from utils import generate_guide_grid, log_frame, record_screen, draw_from_pred, from_variable_to_numpy, monitor_guide, norm_image
from models_dla import init_models
import os
import sys
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import multiprocessing as _mp
from retinanet.encoder import DataEncoder
mp = _mp.get_context('spawn')


def draw_prediction(step, args, output, bboxes, scores, name, save_path):
    if not os.path.isdir(os.path.join(save_path, str(step), name)):
        os.makedirs(os.path.join(save_path, str(step), name))

    if args.use_detection:
        bboxes = bboxes[1:]
        scores = scores[1:]

    s = "step: {}\n".format(step)
    for i in range(args.pred_step):
        img = draw_from_pred(args, from_variable_to_numpy(torch.argmax(output['seg_pred'][0, i+1], 0)))

        if args.use_detection:
            box_list, score_list = bboxes[i], scores[i]
            if box_list.size(0) > 0:
                # detected some bboxes
                for box_id in range(box_list.size(0)):
                    box = box_list[box_id]
                    score = score_list[box_id]
                    s += 'bbox: {} {} {} {} {}\n'.format(box[0], box[1], box[2], box[3], round(score.item(), 3))
                    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
                    # cv2.putText(img, "{}".format(round(score.item(), 3)), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.putText(img, 'OffRoad: %0.2f%%' % round(float(100 * output['offroad_prob'][0, i, 1]), 4), (20, 220), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
        # cv2.putText(img, 'Collision: %0.2f%%' % round(float(100 * output['coll_prob'][0, i, 1]), 4), (20, 230), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
        s += 'Step %d\n' % i
        if args.use_offroad:
            s += 'OffRoad: %0.2f%%\n' % float(100 * output['offroad_prob'][0, i, 1])
        if args.use_collision:
            s += 'Collision: %0.2f%%\n' % float(100 * output['coll_prob'][0, i, 1])
        if args.use_offlane:
            s += 'Offlane: %0.2f%%\n' % float(100 * output['offlane_prob'][0, i, 1])
        # s += 'Distance: %0.2f%%\n' % float(output['dist'][0, i, 0])
        cv2.imwrite(os.path.join(save_path, str(step), name, 'seg%d.png' % (i+1)), img)

    with open(os.path.join(save_path, str(step), name, 'pred_frame.txt'), 'w') as f:
        f.write(s)


def draw_current_frame(args, action, obs, guidance_distri, guide_action, box_list, score_list, save_path, step):
    # draw the visualization for the current frame
    img_save_path = os.path.join(save_path, 'obs.png')
    img = cv2.cvtColor(obs[..., ::-1], cv2.COLOR_BGR2RGB)
    log_file = os.path.join(save_path, 'cur_frame.txt')
    log_file = open(log_file, 'w')
    s = ""

    if args.use_detection:
        box_list = box_list[0]
        score_list = score_list[0]

        if box_list.size(0) > 0:
            # detected some bboxes
            for box_id in range(box_list.size(0)):
                box = box_list[box_id]
                score = score_list[box_id]
                s += "{} {} {} {} {}\n".format(box[0], box[1], box[2], box[3], round(score.item(), 3))
                # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
                # cv2.putText(img, "{}".format(round(score.item(), 3)), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)

    # img = monitor_guide(img, guide_action, guidance_distri)
    s += "Action:Throttle: {} | Steer: {}\n".format(round(action[0], 2), round(action[1], 2))
    log_file.write(s)
    # cv2.putText(img, "Action:Throttle: {} | Steer: {}".format(round(action[0], 2), round(action[1], 2)), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imwrite(img_save_path, img)


def net_infer(args, obs_var, action, net, action_var):
    obs_var = norm_image(obs_var)
    obs_var = obs_var.view(1, 1, 3*args.frame_history_len, args.frame_height, args.frame_width)

    action = torch.from_numpy(action).view(1, args.pred_step, args.num_total_act)
    action = Variable(action.cuda().float(), requires_grad=False)

    net = net.eval()
    with torch.no_grad():
        output = net(obs_var, action, training=False, action_var=action_var)
        if args.use_offroad:
            output['offroad_prob'] = F.softmax(output['offroad_prob'], -1)
        if args.use_collision:
            output['coll_prob'] = F.softmax(output['coll_prob'], -1)
        if args.use_offlane:
            output['offlane_prb'] = F.softmax(output['offlane_prob'], -1)
        if args.SAS:
            output['coll_veh'] = F.softmax(output['coll_veh'], -1)

    return output


def evaluate_policy(args, env):
    guides = generate_guide_grid(args.bin_divide)
    args.checkpoint = args.checkpoint
    net, optimizer, epoch, exploration, num_steps = init_models(args)
    output_path = args.output_path
    for episode in range(100):
        buffer_manager = BufferManager(args)
        action_manager = ActionSampleManager(args, guides)
        action_var = torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, args.frame_history_len - 1, 1).float()

        # initialize environment
        obs_ori, info = env.reset()
        obs = obs_ori.reshape((args.frame_height, 4, args.frame_width, 4, 3)).max(3).max(1)
        info['seg'].resize(args.frame_height, args.frame_width)

        encoder = DataEncoder()

        print('Start episode...')

        for step in range(args.max_eval_step):
            obs_var = buffer_manager.store_frame(obs, info)
            action, guide_action, p = action_manager.sample_action(net=net,
                                                                obs=obs,
                                                                obs_var=obs_var,
                                                                action_var=action_var,
                                                                exploration=exploration,
                                                                step=step,
                                                                explore=False,
                                                                testing=True)
            
            # in the test mode, sample_action outputs the guide_action and action
            # for future pred_step steps, while we only take those for the next frame
            # to execute and store into buffer
            output = net_infer(args, obs_var, action, net, action_var)

            bboxes, labels, scores = [], [], []
            if args.use_detection:
                loc_preds, cls_preds = output['loc_pred'][0].cpu(), output['cls_pred'][0].cpu()
                
                for find in range(loc_preds.size(0)):
                    print(cls_preds[find].sigmoid().max())
                    frame_loc_pred = loc_preds[find]
                    frame_cls_pred = cls_preds[find]
                    pred_bboxes, pred_labels, pred_scores = encoder.decode(frame_loc_pred, frame_cls_pred, input_size=(args.frame_width, args.frame_height))
                    bboxes.append(pred_bboxes)
                    labels.append(pred_labels)
                    scores.append(pred_scores)
            
            draw_prediction(step, args, output, bboxes, scores, 'outcome', '{}/{}'.format(output_path, episode))

            # in the testing mode, $action and $guide_action are for multiple samples, we only pick up the first one
            guide_action = guide_action[0]
            action = action[0]

            obs_ori, reward, done, info = env.step(action)
            obs = obs_ori.reshape((args.frame_height, 4, args.frame_width, 4, 3)).max(3).max(1)
            info['seg'].resize(args.frame_height, args.frame_width)
            draw_current_frame(args, action, obs_ori, p, None, bboxes, scores, os.path.join(output_path, str(episode), str(step)), step)

            print("step: {0} | action [{1:.2f}, {2:.2f}] coll {3} offroad {4} offlane {5} speed {6:.2f} reward {7:.2f}".format(step, action[0], action[1], info['collision'], info['offroad'],info['offlane'], info['speed'], reward))

            action_var = buffer_manager.store_effect(guide_action=guide_action,
                                                    action=action,
                                                    reward=reward,
                                                    done=done,
                                                    info=info)
            if done:
                break
