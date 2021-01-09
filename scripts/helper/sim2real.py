# to test the sim2real performance without finetuning on real-world driving datasets
# a default trained checkpoint from CARLA Town01 is available: https://drive.google.com/file/d/1kyL7XlTHNbl0Zi30T0YBdybHeKIQj_HZ/view?usp=sharing
# some randomly selected sampled from this script are available here: https://drive.google.com/file/d/126qvopTtu8nnly3ORW_B2Z3BumuWkRmv/view?usp=sharing
import torch 
import torch.nn as nn
import sys 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os 

sys.path.append("..")
from models.model import ConvLSTMMulti
import argparse
from args import init_parser, post_processing
import cv2 
from utils import generate_guide_grid, PiecewiseSchedule, monitor_guide, draw_from_pred, from_variable_to_numpy
from utils.util import norm_image
import numpy as np
from actionsampler import ActionSampleManager


parser = argparse.ArgumentParser(description="sim2real")
init_parser(parser)
args = parser.parse_args()
args = post_processing(args)


def draw_output(args, obs, output, action, guidance, guidance_distri, index):
    # draw the visualization for the current frame
    h, w = obs.shape[0], obs.shape[1]
    seg_canvas = np.zeros((h, w*11, 3))
    for i in range(11):
        img = draw_from_pred(args, from_variable_to_numpy(torch.argmax(output['seg_pred'][0, i+1], 0)))
        img = cv2.resize(img, (w, h))
        seg_canvas[:, w*i:w*(i+1)] = img

    cv2.imwrite(os.path.join("../sim2real_vis_kitti/{}_seg.png".format(index)), seg_canvas)
    obs = monitor_guide(obs, guidance, guidance_distri, radius=200)
    # img = monitor_guide(img, guide_action, guidance_distri)
    text = ""
    text += "Throttle: {} | Steer: {}".format(round(action[0].cpu().numpy().item(), 2), round(action[1].cpu().numpy().item(), 2))
    cv2.putText(obs, text, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join("../sim2real_vis_kitti/{}.png".format(index)), obs)


class KittiDataset(Dataset):
    def __init__(self, root, sample_len=3):
        self.root = root  
        self.frame_rate = 30 
        self.data = dict()
        self.data = []
        self.sample_len = sample_len

        for folder in ["image_00", "image_01", "image_02", "image_03"]:
            filelist = os.listdir(os.path.join(self.root, folder, "data"))
            filelist = sorted(filelist)
            for imgname in filelist:
                # self.data.append(cv2.imread(os.path.join(self.root, imgname)))
                imgpath = os.path.join(self.root, folder, "data", imgname)
                self.data.append(imgpath)
            
    def __getitem__(self, index):
        # imgs = [torch.Tensor(cv2.resize(cv2.imread(self.data[i]),  (256, 256))) for i in range(index, index+3)]
        # imgs_ori = [torch.Tensor(cv2.imread(self.data[i])) for i in range(index, index+3)]
        imgs = [torch.Tensor(cv2.resize(cv2.imread(self.data[index]),  (256, 256))) for i in range(index, index+3)]
        imgs_ori = [torch.Tensor(cv2.imread(self.data[index])) for i in range(index, index+3)]
        return (torch.stack(imgs), torch.stack(imgs_ori))

    def __len__(self):
        # return sum([self.data[k]["number"]-5 for k in self.data.keys()])
        return len(self.data) - 5


class BDDDataset(Dataset):
    def __init__(self, root, sample_len=3):
        self.root = root  
        self.frame_rate = 30 
        self.data = dict()
        self.data = []
        self.sample_len = sample_len

        filelist = os.listdir(self.root)
        filelist = sorted(filelist)
        for imgname in filelist:
            # self.data.append(cv2.imread(os.path.join(self.root, imgname)))
            self.data.append(os.path.join(self.root, imgname))
        
    def __getitem__(self, index):
        #imgs = [torch.Tensor(cv2.resize(cv2.imread(self.data[i]),  (256, 256))) for i in range(index, index+3)]
        #imgs_ori = [torch.Tensor(cv2.imread(self.data[i])) for i in range(index, index+3)]
        imgs = [torch.Tensor(cv2.resize(cv2.imread(self.data[index]),  (256, 256))) for i in range(index, index+3)]
        imgs_ori = [torch.Tensor(cv2.imread(self.data[index])) for i in range(index, index+3)]
        return (torch.stack(imgs), torch.stack(imgs_ori))

    def __len__(self):
        # return sum([self.data[k]["number"]-5 for k in self.data.keys()])
        return len(self.data) - 5


def main(dataset_type):
    if dataset_type == "bdd":
        dataset = BDDDataset(root="/home/jinkun/datasets/bdd_val")
    elif dataset_type == "kitti":
        dataset = KittiDataset(root="/home/jinkun/datasets/kitti/2011_09_26_drive_0019_extract")
    else:
        assert(0)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    model = ConvLSTMMulti(args).cuda()
    model.eval()
    state_dict = torch.load("/home/jinkun/git/spc_trained.pt")
    model.load_state_dict(state_dict)

    guides = generate_guide_grid(args.bin_divide)
    actionsampler = ActionSampleManager(args, guides)

    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    for index, data in enumerate(dataloader):
        imgs, imgs_ori = data
        # imgs: sized [batch_size, frame, h, w, 3]
        imgs = torch.transpose(imgs, 2, 4).transpose(3, 4).cuda()
        imgs_ori = torch.transpose(imgs_ori, 2, 4).transpose(3, 4)
        obs_var = imgs.reshape(1, 9, 256, 256).cuda()
        obs = imgs[0, -1].transpose(0, 2).transpose(0, 1).cpu().numpy()
        obs_ori = imgs_ori[0, -1].transpose(0, 2).transpose(0, 1).numpy()
        action_var = torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, args.frame_history_len - 1, 1).float().cuda()
        action, guidance_action, p = actionsampler.sample_action(net=model, obs=obs, obs_var=obs_var, action_var=action_var, exploration=exploration, step=0, testing=True)
        throttle = action[0] * 0.5 + 0.5
        steer = action[1] * 0.4
        obs_var = norm_image(obs_var).unsqueeze(0)
        action = torch.Tensor(action).cuda().unsqueeze(0)
        output = model(obs_var, action, training=False, action_var=action_var)

        action = action[0][0]
        guidance_action = guidance_action[0]
        draw_output(args, obs_ori, output, action, guidance_action, p, index)
        


if __name__ == "__main__":
    main("kitti")
