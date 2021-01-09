import numpy as np
import os

data_src_dir = "datav4"
episodes = os.listdir(data_src_dir)

empty_frame = 0
total_frame = 0

empty_frame_list = open("empty_framev4.txt", 'w')
within_3d_bbox = 0
total_3d_bbox = 0

for episode in episodes:
    episode_path = os.path.join(data_src_dir, episode)
    dir_2d = os.path.join(episode_path, "2d_bbox")
    dir_3d = os.path.join(episode_path, "3d_bbox")
    frames = sorted(os.listdir(dir_2d))
    for frame in frames:
        fpath_2d = os.path.join(dir_2d, frame)
        fpath_3d = os.path.join(dir_3d, frame)
        anno_2d = np.load(fpath_2d)
        anno_3d = np.load(fpath_3d)
        if anno_2d.shape == (1,):
            assert(anno_3d.shape == (1,))

        if anno_3d.shape == (1,):
            assert(anno_2d.shape == (1,))
            empty_frame += 1
            empty_frame_list.write("{}/{}\n".format(episode, frame))
        else:
            num = anno_3d.shape[0]
            total_3d_bbox += num
            at_least_one_within = False
            for ind in range(num):
                bbox3d = anno_3d[ind]
                x_legal = (0 < bbox3d[:,0]).all() and (bbox3d[:,0] < 512).all()
                y_legal = (0 < bbox3d[:,1]).all() and (bbox3d[:,1] < 256).all()
                if x_legal and y_legal:
                    within_3d_bbox += 1
                    at_least_one_within = True
            
            if not at_least_one_within:
                empty_frame += 1
                empty_frame_list.write("{}/{}\n".format(episode, frame))


        total_frame += 1

print("{} / {}".format(empty_frame, total_frame))
print("{} / {}".format(within_3d_bbox, total_3d_bbox))
