import os
from utils.draw import draw_bbox_with_info
import cv2 
import numpy as np
from envs.CARLA.carla_utils import draw_3d_bbox

class MonitorManager():
    # class to manage visulized record from simulator
    def __init__(self, args, path, subdirs, width, height):
        self.args = args
        self.path = path
        self.subdirs = subdirs
        self.width = width
        self.height = height
        self.vis_list = ['offroad', 'offlane', 'collision_other', 'collision_vehicles', 'coll_veh_num', 'collision', 'speed']
        for subdir in self.subdirs:
            subdir_path = os.path.join(self.path, subdir)
            if not os.path.isdir(subdir_path):
                os.makedirs(subdir_path)

        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    def draw_and_save(self, subdir, episode, step, img, info, bg_color=(0,128,128), text_color=(0,0,255), tck=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL):
        bboxes = info['bboxes'] if 'bboxes' in info.keys() else []
        distances = info['distances'] if 'distances' in info.keys() else []
        coll_with = info['coll_with'] if 'coll_with' in info.keys() else []
        bboxes_3d = info["3d_bboxes"] if "3d_bboxes" in info.keys() else []
        
        if self.args.use_detection:
            # visulize the 2d-bboxes
            n_bbox = len(bboxes)
            for ind in range(n_bbox):
                # note that the bbox might be outside the view
                bbox = bboxes[ind]
                dis = distances[ind]
                # is_coll = (ind in coll_with_idx)
                is_coll = coll_with[ind]
                x1, y1, x2, y2 = bbox
                x1d = int(max(x1, 0))
                y1d = int(max(y1, 0))
                x2d = int(min(round(x2+0.5), self.width-1))
                y2d = int(min(round(y2+0.5), self.height-1))
                assert((x1d<x2d) and (y1d<y2d)), "{} {} {} {}".format(x1, y1, x2, y2)
                # the bbox area is inside the view
                text = "{}".format(round(dis,3))
                if is_coll: text = text + " ** COLLISION **"
                bbox_to_draw = [x1d, y1d, x2d, y2d]
                img = draw_bbox_with_info(img, bbox_to_draw, text, font=font)
        
        if self.args.use_3d_detection:
            # visulize the 3d-bboxes
            n_3d_bbox = len(bboxes_3d)
            for ind in range(n_3d_bbox):
                bbox = bboxes_3d[ind]
                bbox = np.array(bbox).reshape(8,2).astype(np.int32)
                x_min = min(bbox[:,0])
                y_min = min(bbox[:,1])
                x_max = max(bbox[:,0])
                y_max = max(bbox[:,1])
                if x_min > self.width-1 or x_max < 0 or y_min > self.height-1 or y_max < 0:
                    continue
                img = draw_3d_bbox(img, bbox)

        self.save(subdir, episode, step, img)

    def save(self, subdir, episode, step, img, subtitle=dict(), color=(0,0,255), tck=1):
        assert(subdir in self.subdirs)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        save_path = os.path.join(self.path, subdir, "%d_%d.png" % (episode, step))
        subtitle_num = 0
        for k in subtitle.keys():
            if k in self.vis_list:
                subtitle_num += 1
                v = subtitle[k]
                if k == 'speed': v = round(v,3)
                text = "{}: {}".format(k, v)
                pos = (12, subtitle_num * 12)
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, tck, cv2.LINE_AA)
        cv2.imwrite(save_path, img)
        
    def merge(self, subdir, episode, max_steps):
        video_dir = os.path.join(self.path, str(episode))
        print("going to merge into : {}".format(video_dir))
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir)
        video_path = os.path.join(video_dir, "{}_{}_{}.mp4".format(episode, subdir, max_steps))
        # fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
        vw = cv2.VideoWriter(video_path, self.fourcc, 24, (self.width, self.height))
        for step in range(max_steps):
            img_path = os.path.join(self.path, subdir, "%d_%d.png" % (episode, step))
            frame = cv2.imread(img_path)
            vw.write(frame)
            # os.remove(img_path)
        vw.release()
        print("record {} video release: {}".format(subdir, video_path))