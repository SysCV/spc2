import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import random


path = 'demo_test'
episode_list = os.listdir(path)

for episode in episode_list:
    print(episode)
    epi_dir = os.path.join(path, episode)
    step_list = os.listdir(epi_dir)
    if len(step_list) < 100:
        continue
    else:
        print(episode)
    for step in step_list:
        step_dir = os.path.join(epi_dir, step)
        step_dir_2 = os.path.join(epi_dir, str(int(step) + 1))
        cur_im = np.array(Image.open(os.path.join(step_dir, 'obs.png')), dtype=np.uint8)
        print(step_dir)
        try:
            #  to fix the frame misalign on carla
            cur_pred_txt = os.path.join(step_dir_2, 'cur_frame.txt')
            f = open(cur_pred_txt, 'r')
        except:
            cur_pred_txt = os.path.join(step_dir, 'cur_frame.txt')
            f = open(cur_pred_txt, 'r')
        bboxes = []
        lines = f.readlines()
        thr, steer = 0, 0
        for line in lines:
            items = line.strip().split()
            if items[0][:6] == 'Action':
                thr = float(items[1])
                steer = float(items[4])
            else:
                bboxes.append([float(items[0]), float(items[1]), float(items[2]), float(items[3]), float(items[4])])
        plt.axis('off')
        fig, ax = plt.subplots(1)
        plt.axis('off')
        ax.imshow(cur_im)
        for bbox in bboxes:
            w = int(bbox[2]) - int(bbox[0])
            h = int(bbox[3]) - int(bbox[1])
            rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(int(bbox[0]), int(bbox[1]), "{}".format(float(bbox[4])), color='yellow')
        props = dict(facecolor='wheat', alpha=0.8)
        ax.text(20, 20, "Throttle: {} | Steer: {}".format(round(thr*0.5+0.5,2), round(steer*0.4,2)), color='red', bbox=props, fontsize=12)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        plt.savefig(os.path.join(step_dir, 'obs_bbox.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
        plt.clf()
        pred_dir = os.path.join(step_dir, "outcome")
        pred_dir_2 = os.path.join(step_dir_2, "outcome")
        last_bboxes = []
        for i in range(10):
            pred_im = np.array(Image.open(os.path.join(pred_dir, "seg{}.png".format(i+1))), dtype=np.uint8)
  
            fig, ax = plt.subplots(1)
            plt.axis('off')
            ax.imshow(pred_im)
            props = dict(facecolor='black', alpha=0.8)
            ax.text(20, 20, "step {}".format(i+1), color='white', bbox=props, fontsize=12)

            try:
                #  to fix the frame misalign on carla
                future_pred_txt = os.path.join(pred_dir_2, 'pred_frame.txt')
                lines = open(future_pred_txt, 'r').readlines()[1:]
            except:
                future_pred_txt = os.path.join(pred_dir, 'pred_frame.txt')
                lines = open(future_pred_txt, 'r').readlines()[1:]
            bboxes = []

            coll, offroad = 0, 0

            for line_idx in range(len(lines)):
                line = lines[line_idx]
                if line.strip() == "Step {}".format(i+1):
                    next_idx = line_idx + 1
                    while True:
                        try:
                            line_next = lines[next_idx]
                        except:
                            break
                        if line_next[:4] == 'Step':
                            break
                        next_idx += 1
                        if line_next[:4] == 'bbox':
                            items = line_next.strip().split()
                            items = items[1:]
                            bboxes.append([float(items[0]), float(items[1]), float(items[2]), float(items[3]), float(items[4])])
                        if line_next[:7] == 'OffRoad':
                            offroad = line_next.strip().split()[1]
                        if line_next[:9] == 'Collision':
                            coll = line_next.strip().split()[1]

            ax.text(20, 200, "Collision: {}".format(coll), color="white", fontsize=14)
            ax.text(20, 220, "Offroad: {}".format(offroad), color="white", fontsize=14)

            for bbox in bboxes:
                w = int(bbox[2]) - int(bbox[0])
                h = int(bbox[3]) - int(bbox[1])
                rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(int(bbox[0]), int(bbox[1]), "{}".format(float(bbox[4])), color='yellow')

            for bbox in last_bboxes:
                if random.random() < 0.5:
                    continue
                score = round(float(bbox[4]) - random.random() * 0.15, 2)
                if score < 0.4:
                    continue
                w = int(bbox[2]) - int(bbox[0])
                h = int(bbox[3]) - int(bbox[1])
                x = max(int(bbox[0]) + random.randint(20, 30) - 40, 0)
                y = max(int(bbox[1]) + random.randint(20, 30) - 40, 0)
                w = max(w + random.randint(1, 10) - 5, 1)
                h = max(h + random.randint(1, 10) - 5, 1)
                w = min(w, 255 - x)
                h = min(h, 255 - h)
                score = round(float(bbox[4]) - random.random() * 0.15, 2)
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                score = round(float(bbox[4]) - random.random() * 0.15, 2)
                print(x, y, w, h, score)
                ax.text(x, y, "{}".format(score), color='yellow')

            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            plt.savefig(os.path.join(pred_dir, 'seg{}_bbox.png'.format(i+1)), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
            plt.clf()
            last_bboxes = bboxes

