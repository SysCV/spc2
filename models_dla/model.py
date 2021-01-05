import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
from models.DLASeg import DLASeg
from models.convLSTM import convLSTM
from models.end_layer import end_layer
from utils import PiecewiseSchedule, tile, tile_first, load_model
from models.retinanet import RetinaNet, RetinaNet_FPN, FPN50
import torch.nn.init as init
import math


class ConvLSTMNet(nn.Module):
    def __init__(self, args):
        super(ConvLSTMNet, self).__init__()
        self.args = args

        # Feature extraction and prediction
        self.dlaseg = DLASeg(args, down_ratio=4)
        self.feature_map_predictor = convLSTM()

        # prepare the feature map to output detection results
        self.detector_cur = RetinaNet()
        self.detector_pred = RetinaNet()
        
        self.conv1 = self.make_seq(256, 256)
        self.conv2 = self.make_seq(128, 256)
        self.conv3 = self.make_seq(64, 256)
        self.conv4 = self.make_seq(64, 256)
        self.conv5 = self.make_seq(64, 256)
        # self.conv1 = nn.Conv2d(256, 64, kernel_size=2, stride=2, bias=True)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=2, stride=2, bias=True)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=True)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=True)
        # self.conv5 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=True)

        # Information prediction
        self.guide_layer = end_layer(args, args.classes, int(np.prod(args.bin_divide)))
        self.coll_layer = end_layer(args, args.classes, 2)
        self.coll_vehicle_layer = end_layer(args, args.classes, 2)
        self.coll_other_layer = end_layer(args, args.classes, 2)
        self.offroad_layer = end_layer(args, args.classes, 2)
        self.offlane_layer = end_layer(args, args.classes, 2)
        self.speed_layer = end_layer(args, args.classes * args.frame_history_len, 1)

    def make_seq(self, in_channel, out_channel, kz=2, stride=2, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kz, stride=stride, bias=bias),
            nn.GroupNorm(out_channel, out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, action, with_encode=False, hidden=None, cell=None, training=True, action_var=None, only_event=False):
        output_dict = dict()

        if not with_encode:
            x, hidden, output_dict['seg_current'], out_fms = self.get_feature(x)
            fm1, fm2, fm3, fm4 = self.conv1(out_fms[0]), self.conv2(out_fms[1]), self.conv3(out_fms[2]), self.conv4(out_fms[3])
            fm0 = self.conv5(fm1)

            if only_event == False:
                loc_preds, cls_preds, colls_with_preds = self.detector_cur([fm4, fm3, fm2, fm1, fm0])
                output_dict['loc_current'] = loc_preds
                output_dict['cls_current'] = cls_preds
                output_dict['colls_with_current'] = colls_with_preds
            
            if torch.cuda.is_available():
                action_var = action_var.cuda()

            x = tile_first(x, action_var)

        x[-1] = tile(x[-1], action)

        hx = self.feature_map_predictor(x)  
        rx, output_dict['seg_pred'], out_fms = self.dlaseg.infer(hx)

        if only_event == False:
            fm1, fm2, fm3, fm4 = self.conv1(out_fms[0]), self.conv2(out_fms[1]), self.conv3(out_fms[2]), self.conv4(out_fms[3])
            fm0 = self.conv5(fm1)

            loc_preds, cls_preds, colls_with_preds = self.detector_pred([fm4, fm3, fm2, fm1, fm0])
            output_dict['loc_pred'] = loc_preds
            output_dict['cls_pred'] = cls_preds
            output_dict['colls_with_prob'] = colls_with_preds

        nx_feature_enc = x[1:] + [hx]
        hidden = torch.cat([hidden[:, self.args.classes:, :, :], rx], dim=1)

        output_dict['coll_prob'] = self.coll_layer(rx.detach())
        output_dict['coll_other_prob'] = self.coll_other_layer(rx.detach())
        output_dict['coll_vehicles_prob'] = self.coll_vehicle_layer(rx.detach())
        output_dict['offroad_prob'] = self.offroad_layer(rx.detach())
        output_dict['offlane_prob'] = self.offlane_layer(rx.detach())
        output_dict['speed'] = self.speed_layer(hidden.detach())

        return output_dict, nx_feature_enc, hidden, None

    def get_feature(self, x, train=True):
        batch_size, frame_history_len, height, width = x.size()
        frame_history_len = int(frame_history_len / 3)
        res = []
        hidden = []

        if train:
            for i in range(frame_history_len):
                this_img = x[:, i*3:(i+1)*3, :, :]
                xx, rx, y, out_fms = self.dlaseg(this_img)
                res.append(xx)
                hidden.append(rx)
            hidden = torch.cat(hidden, dim=1)
            return res, hidden, y, out_fms
        else:
            for i in range(frame_history_len):
                this_img = x[:, i*3:(i+1)*3, :, :]
                xx = self.dlaseg(this_img, train=train)
                res.append(xx)
            return res


class ConvLSTMMulti(nn.Module):
    def __init__(self, args):
        super(ConvLSTMMulti, self).__init__()
        self.args = args
        self.conv_lstm = ConvLSTMNet(self.args)
        self.retinanet = RetinaNet_FPN()
        self.init_retinanet()
    
    def init_retinanet(self):
        d = torch.load('./model/resnet50.pth')
        fpn = FPN50()
        dd = fpn.state_dict()
        for k in d.keys():
            if not k[:2] == 'fc':
                dd[k] = d[k]
            else:
                print('skip the fc layers: {}'.format(k))

        for m in self.retinanet.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        pi = 0.01
        init.constant_(self.retinanet.cls_head[-1].bias, -math.log((1-pi)/pi))
        self.retinanet.fpn.load_state_dict(dd)

    def get_feature(self, x, next_obs=False):
        x = x.contiguous()
        if next_obs:
            batch_size, pred_step, frame_history_len, frame_height, frame_width = x.size()
            frame_history_len = int(frame_history_len / 3)
            x = x.view(batch_size * pred_step, 3 * frame_history_len, frame_height, frame_width)
            x = self.conv_lstm.get_feature(x)[1]
            x = x.contiguous().view(batch_size, pred_step, self.args.classes * frame_history_len, frame_height, frame_width)
            return x
        else:
            xx, x, y = self.conv_lstm.get_feature(x[:, 0, :, :, :])
            return xx, x, y

    def guide_action(self, x):
        _, enc, seg, _ = self.conv_lstm.dlaseg(x)
        logit = self.conv_lstm.guide_layer(enc.detach())
        return logit

    # TODO: refactor this function
    def forward(self, imgs, actions=None, hidden=None, cell=None, get_feature=False, training=True, function='', action_var=None, next_obs=False):
        # create dictionary to store outputs
        retinanet_loc_preds, retinanet_cls_preds = self.retinanet(imgs[:, 0, 6:9, :, :])

        final_dict = dict()
        final_dict['loc_preds_rt'] = retinanet_loc_preds
        final_dict['cls_preds_rt'] = retinanet_cls_preds

        return final_dict
        
        if function == 'guide_action':
            return self.guide_action(imgs)

        only_event = (function == "pred_event")

        # TODO: simplify the line 162
        batch_size, num_step, c, w, h = int(imgs.size()[0]), int(imgs.size()[1]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        
        output_dict, pred, hidden, cell = self.conv_lstm(imgs[:, 0, :, :, :].squeeze(1), actions[:, 0, :].squeeze(1), hidden=hidden, cell=cell, training=training, action_var=action_var, only_event=only_event)

        for key in output_dict.keys():
            final_dict[key] = [output_dict[key]]

        # combine the result for the current frame and the next future frame
        if only_event == False:
            final_dict['seg_pred'] = [output_dict['seg_current'], output_dict['seg_pred']]
            final_dict['loc_pred'] = [output_dict['loc_current'], output_dict['loc_pred']]
            final_dict['cls_pred'] = [output_dict['cls_current'], output_dict['cls_pred']]
            final_dict['colls_with_prob'] = [output_dict['colls_with_current'], output_dict['colls_with_prob']]

        # keep predicting for following future frames and append prediction into the dict
        for i in range(1, self.args.pred_step):
            output_dict, pred, hidden, cell = self.conv_lstm(pred, actions[:, i, :], with_encode=True, hidden=hidden, cell=cell, training=training, action_var=None, only_event=only_event)
            for key in output_dict.keys():
                final_dict[key].append(output_dict[key])

        for key in final_dict.keys():
            if key == 'loc_preds_rt' or key == 'cls_preds_rt':
                continue
            final_dict[key] = torch.stack(final_dict[key], dim=1)
        
        return final_dict


def init_models(args):
    train_net = ConvLSTMMulti(args)
    for param in train_net.parameters():
        param.requires_grad = True
    train_net.train()

    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    train_net, epoch = load_model(args.save_path, train_net, resume=args.resume)
    net.load_state_dict(train_net.state_dict())

    if torch.cuda.is_available():
        train_net = train_net.cuda()
        net = net.cuda()
        if args.data_parallel:
            train_net = torch.nn.DataParallel(train_net)
            net = torch.nn.DataParallel(net)
    
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
        except:
            num_imgs_start = 0
    else:
        num_imgs_start = 0

    return train_net, net, optimizer, epoch, exploration, num_imgs_start
