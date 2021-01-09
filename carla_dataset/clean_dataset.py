import os
import numpy as np
from shutil import copyfile

def count_empty_frame(data_src_dir="data", record_txt="empty_frame4.txt"):
    # search which frames having no vehicle instance bbox annotations
    episodes = os.listdir(data_src_dir)
    empty_frame = 0
    total_frame = 0
    empty_frame_list = open(record_txt, 'w')
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
'''
print("{} / {}".format(empty_frame, total_frame))
print("{} / {}".format(within_3d_bbox, total_3d_bbox))
empty_frame_list = open("empty_frame.txt", 'r')
empty_frame_dict = {}

lines = empty_frame_list.readlines()
for line in lines:
    items = line.split("/")
    episode = items[0]
    frame = items[1].split('.')[0]
    if episode not in empty_frame_dict.keys():
        empty_frame_dict[episode] = []
        empty_frame_dict[episode].append(frame)
    else:
        empty_frame_dict[episode].append(frame)

for episode in empty_frame_dict.keys():
    origin_path = os.path.join("data", episode, "obs")
    frame_num = len(os.listdir(origin_path))
    total_frame_indices = [str(i+1) for i in range(frame_num)]
    print(total_frame_indices)
'''

def remove_empty_frames(empty_list, data_dir="datav4", episode_len_thr=40):
    # to remove bbox-free frames and maintain the episode length at least xxx
    # too short episodes will be deleted
    f = open(empty_list, 'r')
    f_res = open("non-empty-video-v4.txt", 'w')
    episode_list = os.listdir(data_dir)
    lines = f.readlines()
    empty_dict = dict()
    for episode in episode_list:
        frame_num = len(os.listdir(os.path.join(data_dir, episode, "obs")))
        frame_list = [str(i+1) for i in range(frame_num)]
        empty_dict[episode] = frame_list

    for line in lines:
        episode = line.split("/")[0]
        frame = line.split("/")[1].split(".")[0]
        empty_dict[episode].remove(frame)

    for episode in empty_dict.keys():
        segmented_list = []
        cur_segment = []
        raw_indices = empty_dict[episode]
        for i in range(len(raw_indices)):
            if i == len(raw_indices) - 1:
                if int(raw_indices[i]) == int(raw_indices[i-1]) - 1:
                    cur_segment.append(raw_indices[i])
                
                segmented_list.append(cur_segment)  
            else:
                if int(raw_indices[i+1]) == int(raw_indices[i]) + 1:
                    cur_segment.append(raw_indices[i])
                else:
                    cur_segment.append(raw_indices[i])
                    segmented_list.append(cur_segment)
                    cur_segment = []
        empty_dict[episode] = segmented_list

    episode_list = sorted(empty_dict.keys())
    for episode in episode_list:
        segmented_list = empty_dict[episode]
        print("{}: {}".format(episode, len(segmented_list)))
        for segment in segmented_list:
            print("{} - {}".format(segment[0], segment[-1]))
            if int(segment[-1]) - int(segment[0]) > episode_len_thr:
                f_res.write("{}: {} - {}\n".format(episode, segment[0], segment[-1]))


def build_new_dataset(non_empty_file, src_dir, dst_dir):
    # function to copy non-empty frames of length longer than a threshold to a new directory
    new_episode_index = 1
    non_empty_list = open(non_empty_file)
    lines = non_empty_list.readlines()
    subdirs = ["2d_bbox", "3d_bbox", "center", "dimensions", "depth", "obs", "seg", "orientations"]
    
    def create_subdir(root_path, sub_dir):
        subpath = os.path.join(root_path, sub_dir)
        if not os.path.isdir(subpath):
            os.makedirs(subpath)
        return subpath

    def copy_to_new(old_dir, new_dir, subdir, old_find, new_find, suffix):
        oldpath = os.path.join(old_dir, subdir, "{}.{}".format(old_find, suffix))
        new_path = os.path.join(new_dir, subdir, "{}.{}".format(new_find, suffix))
        copyfile(oldpath, new_path)

    def create_new_subdirs(root_path, subdir_list):
        for subdir in subdir_list:
            subdir_path = os.path.join(root_path, subdir)
            if not os.path.isdir(subdir_path):
                os.makedirs(subdir_path)

    for line in lines:
        items = line.split()
        episode = items[0][:-1]
        start_frame = int(items[1])
        end_frame = int(items[3])
        ori_episode_path = os.path.join(src_dir, episode)
        new_episode_path = os.path.join(dst_dir, str(new_episode_index))
        if not os.path.isdir(new_episode_path):
            os.makedirs(new_episode_path)

        f_state = open(os.path.join(new_episode_path, "state.txt"), 'w')
        f_action = open(os.path.join(new_episode_path, "action.txt"), 'w')
        f_coll = open(os.path.join(new_episode_path, "coll_withs.txt"), 'w')
        state_lines_old = open(os.path.join(ori_episode_path, 'state.txt'), 'r').readlines()
        action_lines_old = open(os.path.join(ori_episode_path, 'action.txt'), 'r').readlines()
        coll_lines_old = open(os.path.join(ori_episode_path, 'coll_withs.txt'), 'r').readlines()

        create_new_subdirs(new_episode_path, subdirs)
        new_frame_index = 1
        for old_find in range(start_frame, end_frame+1):
            state = state_lines_old[old_find-1]
            action = action_lines_old[old_find-1]
            coll = coll_lines_old[old_find-1]

            f_state.write(state)
            f_action.write(action)
            f_coll.write(coll)

            for subdir in subdirs:
                suffix = "jpg" if subdir == "obs" else "npy" 
                if subdir == "center":
                    continue
                copy_to_new(ori_episode_path, new_episode_path, subdir, old_find, new_frame_index, suffix)

            new_frame_index += 1

        new_episode_index += 1


def main():
    remove_empty_frames("empty_framev4.txt")
    # build_new_dataset("non-empty-video.txt", "data", "data_clean")

if __name__ == "__main__":
    main()
