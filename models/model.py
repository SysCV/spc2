import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
from models.convLSTM import convLSTM
from models.end_layer import end_layer
from utils import PiecewiseSchedule, tile, tile_first, load_model
from models.retinanet import FPN50, RetinaNet_Header
import torch.nn.init as init
import math
import torch.nn.functional as F


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest", align_corners=None):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class ConvLSTMNet(nn.Module):
    def __init__(self, args):
        super(ConvLSTMNet, self).__init__()
        self.args = args
        # Feature extraction and prediction
        self.fpn = FPN50()
        self.feature_map_predictor = convLSTM(self.args)

        # since the distribution different between feature maps from real RGB images
        # and those from LSTM, we compromisingly use two detectors, each for one case
        self.detector = RetinaNet_Header()

        # be default we use an 4x upsampling to recover feature map into original image scale
        self.up = nn.Sequential(
            # 1. conv the channel number to the number of semantic segmentation labels
            nn.Conv2d(256, self.args.classes, kernel_size=1, stride=1, padding=0, bias=True),
            # 2. upsample the feature map into the original image scale
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(args.classes, args.classes, 3, padding=1),
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(args.classes, args.classes, 5, padding=2),
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(args.classes, args.classes, 5, padding=2)
        )

        self.depth_head = nn.Sequential(
            # this up-sampling head is for depth estimation
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1, 1, 3, padding=1),
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1, 1, 5, padding=2),
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1, 1, 5, padding=2)
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        # Information prediction
        self.guide_layer = end_layer(args, args.classes, int(np.prod(args.bin_divide)))
        self.coll_layer = end_layer(args, args.classes, 2)
        # self.coll_vehicle_layer = end_layer(args, args.classes, 2)
        # self.coll_other_layer = end_layer(args, args.classes, 2)
        self.offroad_layer = end_layer(args, args.classes, 2)
        self.offlane_layer = end_layer(args, args.classes, 2)
        self.speed_layer = end_layer(args, args.classes * args.frame_history_len, 1)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.fpn.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        for layer in self.detector.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def fm_infer(self, fms):
        feat = self.up(fms[0])
        feat_depth = self.depth_head(fms[0])
        seg = self.logsoftmax(feat)
        hidden = self.softmax(feat)
        return seg, hidden, feat_depth

    def act(self, x):
        fms = self.fpn(x)
        _, hidden, _ = self.fm_infer(fms)
        logit = self.guide_layer(hidden.detach())
        return logit

    def get_feature(self, x, train=True):
        # one forward for a histroy frame sequence
        fnum = x.size(1)
        assert(fnum % 3 == 0) 
        fnum = int(fnum / 3)

        fms_seq, hidden_seq = [], []
        for fidx in range(fnum):
            frame = x[:, 3*fidx: 3*(fidx+1), :, :]
            fms = self.fpn(frame)
            '''
            shape of fms:
                fms[0]: B x 256 x (H/4) x (W/4)
                fms[1]: B x 256 x (H/8) x (W/8)
                fms[2]: B x 256 x (H/16) x (W/16)
                fms[3]: B x 256 x (H/32) x (W/32)
                fms[4]: B x 256 x (H/64) x (W/64)
            '''
            seg, hidden, depth = self.fm_infer(fms)
            fms_seq.append(fms)
            hidden_seq.append(hidden)
        hidden_seq = torch.cat(hidden_seq, dim=1)
        return fms_seq, hidden_seq, seg, depth, fms

    '''
    def predict_next(self, fms_seq, pred_fms, hidden):
        # predict items-of-interest on predicted feature maps
        output_dict = dict()
        output_dict['seg_pred'], rx, output_dict['depth_pred'] = self.fm_infer(pred_fms)
        if self.args.use_detection:
            output_dict['loc_pred'], output_dict['cls_pred'], output_dict['colls_with_prob'], output_dict["residual_pred"], output_dict["conf_pred"], output_dict["dim_pred"], output_dict["center_pred"] = self.detector(pred_fms)

        nx_feature_enc = fms_seq[1:] + [pred_fms]
        hidden = torch.cat([hidden[:, self.args.classes:, :, :], rx], dim=1)

        if self.args.use_collision:
            output_dict['coll_prob'] = self.coll_layer(rx.detach())
            # output_dict['coll_other_prob'] = self.coll_other_layer(rx.detach())
            # output_dict['coll_vehicles_prob'] = self.coll_vehicle_layer(rx.detach())
        if self.args.use_offroad:
            output_dict['offroad_prob'] = self.offroad_layer(rx.detach())
        if self.args.use_offlane:
            output_dict['offlane_prob'] = self.offlane_layer(rx.detach())
        if self.args.use_speed:
            output_dict['speed'] = self.speed_layer(hidden.detach())

        return output_dict, nx_feature_enc, hidden
    '''

    def forward_next_step(self, fms_seq, action, with_encode=False, hidden=None, cell=None, training=True, action_var=None):
        # given the predicted feature maps for the next frame, do:
        # 1. detect vehicles on the feature maps
        # 2. infer semantic segmentation on the feature maps
        # 3. infer feature maps for the following frame
        # 4. predct events on the feature maps
        output_dict = dict()
        fms_seq[-1] = tile(fms_seq[-1], action)
        pred_fms = self.feature_map_predictor(fms_seq)

        # output_dict, nx_feature_enc, hidden = self.predict_next(fms_seq, pred_fms, hidden)
        output_dict['seg_pred'], rx, output_dict['depth_pred'] = self.fm_infer(pred_fms)
        if self.args.use_detection:
            output_dict['loc_pred'], output_dict['cls_pred'], output_dict['colls_with_prob'], output_dict["residual_pred"], output_dict["conf_pred"], output_dict["dim_pred"], output_dict["center_pred"] = self.detector(pred_fms)

        nx_feature_enc = fms_seq[1:] + [pred_fms]
        hidden = torch.cat([hidden[:, self.args.classes:, :, :], rx], dim=1)

        if self.args.use_collision:
            output_dict['coll_prob'] = self.coll_layer(rx.detach())
            # output_dict['coll_other_prob'] = self.coll_other_layer(rx.detach())
            # output_dict['coll_vehicles_prob'] = self.coll_vehicle_layer(rx.detach())
        if self.args.use_offroad:
            output_dict['offroad_prob'] = self.offroad_layer(rx.detach())
        if self.args.use_offlane:
            output_dict['offlane_prob'] = self.offlane_layer(rx.detach())
        if self.args.use_speed:
            output_dict['speed'] = self.speed_layer(hidden.detach())

        return output_dict, nx_feature_enc, hidden, None

    def forward(self, x, action, with_encode=False, hidden=None, cell=None, training=True, action_var=None):
        # given the RGB observations of current frame and history frames, do:
        # 1. detect vehicles on the current frame & next frame
        # 2. infer semantic segmentation on the current frame & next frame
        # 3. infer feature maps for the next frame with LSTM
        # 4. predict events on the next frame
        output_dict = dict()
        fms_seq, hidden_seq, last_frame_seg, last_frame_depth, last_frame_fms = self.get_feature(x)

        output_dict['seg_current'] = last_frame_seg
        output_dict["depth_current"] = last_frame_depth

        if self.args.use_detection:
            output_dict['loc_current'], output_dict['cls_current'], output_dict['coll_with_current'], output_dict["residual_current"], output_dict["conf_current"], output_dict["dim_current"], output_dict["center_current"] = self.detector(last_frame_fms)
        
        fms_seq = tile_first(fms_seq, action_var)
        output_dict_future, nx_feature_enc, hidden, _ = self.forward_next_step(fms_seq, action, hidden=hidden_seq)

        output_dict_all = dict(output_dict, **output_dict_future)  

        return output_dict_all, nx_feature_enc, hidden, None


class ConvLSTMMulti(nn.Module):
    def __init__(self, args):
        super(ConvLSTMMulti, self).__init__()
        self.args = args
        self.conv_lstm = ConvLSTMNet(self.args)

    def guide_action(self, x):
        _, enc, seg, _ = self.conv_lstm.dlaseg(x)
        logit = self.conv_lstm.guide_layer(enc.detach())
        return logit

    # TODO: refactor this function
    def forward(self, imgs, actions=None, hidden=None, cell=None, get_feature=False, training=True, action_var=None, next_obs=False, action_only=False):
        # create dictionary to store outputs
        # retinanet_loc_preds, retinanet_cls_preds = self.retinanet(imgs[:, 0, 6:9, :, :])
        final_dict = dict()

        if action_only:
            return self.conv_lstm.act(imgs)

        output_dict, pred, hidden, cell = self.conv_lstm(imgs[:, 0, :, :, :], actions[:, 0, :], hidden=hidden, cell=cell, training=training, action_var=action_var)
        
        for key in output_dict.keys():
            final_dict[key] = [output_dict[key]]

        # combine the result for the current frame and the next future frame
        final_dict['seg_pred'] = [output_dict['seg_current'], output_dict['seg_pred']]
        final_dict["depth_pred"] = [output_dict["depth_current"], output_dict["depth_pred"]]

        if self.args.use_detection:
            final_dict['loc_pred'] = [output_dict['loc_current'], output_dict['loc_pred']]
            final_dict['cls_pred'] = [output_dict['cls_current'], output_dict['cls_pred']]
            final_dict['colls_with_prob'] = [output_dict['coll_with_current'], output_dict['colls_with_prob']]
            '''
            final_dict['residual_pred'] = [output_dict['residual_current'], output_dict['residual_pred']]
            final_dict['conf_pred'] = [output_dict['conf_current'], output_dict['conf_pred']]
            final_dict['dim_pred'] = [output_dict['dim_current'], output_dict['dim_pred']]
            final_dict['center_pred'] = [output_dict['center_current'], output_dict['center_pred']]
            '''

        # keep predicting for following future frames and append prediction into the dict
        for i in range(1, self.args.pred_step):
            output_dict, pred, hidden, cell = self.conv_lstm.forward_next_step(pred, actions[:, i, :], with_encode=True, hidden=hidden, cell=cell, training=training, action_var=None)
            for key in output_dict.keys():
                final_dict[key].append(output_dict[key])

        for key in final_dict.keys():
            final_dict[key] = torch.stack(final_dict[key], dim=1)
        
        return final_dict


def init_models(args):
    train_net = ConvLSTMMulti(args)
    for param in train_net.parameters():
        param.requires_grad = True

    if not args.eval:
        train_net.train()

    train_net, epoch = load_model(args, args.save_path, train_net, resume=args.resume)
    
    '''
    if not args.resume and not args.eval:
        pretrain_net = torch.load(args.pretrain_model)
        try:
            train_net.load_state_dict(pretrain_net)
        except:
            train_net.load_state_dict(pretrain_net, strict=False)
            print("strict load checkpoint {} failed. turn to non-strict loading mode".format(pretrain_net))
        train_net.conv_lstm.freeze_bn()
    '''

    if torch.cuda.is_available():
        train_net = train_net.cuda()
        if args.data_parallel:
            train_net = torch.nn.DataParallel(train_net)
    
    if args.optim == 'Adam':
        optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(train_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        assert(0)

    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    if args.resume:
        try:
            if args.env == 'torcs':
                num_imgs_start = max(int(open(os.path.join(args.save_path, 'log_train_torcs.txt')).readlines()[-1].split(' ')[1]) - 1000, 0)
            elif 'carla' in args.env:
                num_imgs_start = int(open(os.path.join(args.save_path, 'log_train_{}.txt'.format(args.env))).readlines()[-1].split(' ')[3])
            else:
                num_imgs_start = 0 # TODO: to be fixed
        except:
            num_imgs_start = 0
    else:
        num_imgs_start = 0

    return train_net, optimizer, epoch, exploration, num_imgs_start
