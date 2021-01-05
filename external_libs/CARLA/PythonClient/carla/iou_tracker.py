import os
import numpy as np

class ins_buffer(object):
    # class to hold instance from the last frame
    def __init__(self, frame_num):
        self.ins_last_frame = {}
        self.frame_num = frame_num
        self.ins_num = 0
    
    def insert(self, ind, area):
        self.ins_last_frame[ind] = area
        self.ins_num = len(self.ins_last_frame.keys())

    def cal_ious(self, masks):
        # masks: predicted M masks
        # refs: reference N masks 
        # return: the iou of each mask referring to each reference: M x N
        iou_dict = dict()
        for ins_id in self.ins_last_frame.keys():
            # ins_id = i + 1
            iou_dict[ins_id] = []
            ref_mask = self.ins_last_frame[ins_id]
            for i in range(len(masks)):
                # if i not in iou_dict.keys():
                #     iou_dict[i] = {}
                mask = masks[i]
                mask_flatten = mask.reshape(-1)
                ref_mask_flatten = ref_mask.reshape(-1)
                iou = jaccard_score(ref_mask_flatten, mask_flatten)
                iou_dict[ins_id].append(iou)
        # note: the score in the iou_dict is ordered
        return self.compare_update(iou_dict, masks)

    def compare_update(self, iou_dict, masks):
        keys = iou_dict.keys()
        res_dict = {}
        max_ins_index = max(keys)
        used_indices = [-1 for _ in range(len(masks))]
        used_ious = [-1 for _ in range(len(masks))]
        for ins_id in keys:
            ious = np.array(iou_dict[ins_id])
            rank_indices = np.argsort(-ious)
            res_dict[ins_id] = {}
            res_dict[ins_id]["score"] = ious
            # res_dict[ins_id]["rank"] = rank_indices
            hit_index = rank_indices[0]
            hit_mask = masks[hit_index]
            if used_indices[hit_index] == -1:
                self.ins_last_frame[ins_id] = hit_mask
                used_ious[hit_index] = ious[hit_index]
                used_indices[hit_index] = ins_id
                res_dict[ins_id]["rank"] = rank_indices
            else:
                if ious[hit_index] > used_ious[hit_index]:
                    prev_ins_id = used_indices[hit_index]
                    del self.ins_last_frame[prev_ins_id]
                    del res_dict[prev_ins_id]
                    used_ious[hit_index] = ious[hit_index]
                    used_indices[hit_index] = ins_id
                    self.ins_last_frame[ins_id] = hit_mask
                    self.ins_num -= 1
                    res_dict[ins_id]["rank"] = rank_indices
                else:
                    self.ins_num -= 1
                    # res_dict[ins_id]["rank"] = [-1]
                    del self.ins_last_frame[ins_id]
                    del res_dict[ins_id]
        return res_dict


import shutil

seq_id_list_total = []
with open(test_id_list) as f:
    lines = f.readlines()
    for line in lines:
        seq_id_list_total.append(line.strip())


def track(start_idx, end_idx):
    seq_id_list = seq_id_list_total[start_idx: end_idx]
    for seq_id in seq_id_list:
        seq_path = os.path.join(anno_path, seq_id)
        fframe_path = os.path.join(seq_path, "00000.png")
        seq_jpeg_path = os.path.join(jpeg_path, seq_id)
        frame_list = os.listdir(seq_jpeg_path)
        frame_num = len(frame_list)
        ref_im = Image.open(fframe_path)
        ref_im = np.asarray(ref_im)
        ibuffer = ins_buffer(frame_num)
        ins_num = np.unique(ref_im).size - 1

        # dst_first_path = os.path.join(dst_path, seq_id, "00000.png")
        # shutil.copyfile(fframe_path, dst_first_path)

        for ins_id in range(1, ins_num+1):
            canvas = np.zeros([ref_im.shape[0], ref_im.shape[1]])
            ins_mask = np.where(ref_im==ins_id)
            canvas[ins_mask] = 1
            ibuffer.insert(ins_id, canvas)

        for frame_ind in range(1, frame_num):
            previous_path = os.path.join(src_path, seq_id, "%05d" % (frame_ind-1))
            pred_path = os.path.join(src_path, seq_id, "%05d" % frame_ind)
            preds = []
            missing = False
            try:
                pred_list = os.listdir(pred_path)
            except:
                pred_list = os.listdir(previous_path)
                missing = True

            for pred in pred_list:
                if not missing:
                    pp_path = os.path.join(pred_path, pred)
                else:
                    pp_path = os.path.join(previous_path, pred)
                im = Image.open(pp_path)
                im = np.asarray(im)
                im.flags.writeable = True
                im[np.where(im<20)] = 0
                im[np.where(im>230)] = 1
                preds.append(im)

            ins_dict = ibuffer.cal_ious(preds)
            canvas = np.zeros([ref_im.shape[0], ref_im.shape[1]])
            for ins_id in ins_dict.keys():
                # pdb.set_trace()
                hit_index = ins_dict[ins_id]["rank"][0]
                if hit_index == -1:
                    continue
                hit_mask = preds[hit_index]
                mask = np.where(hit_mask == 1)
                canvas[mask] = ins_id

            canvas = canvas.astype(np.uint8)
            canvas = Image.fromarray(canvas, mode='P')
            canvas.putpalette(template_palette)
            save_path = os.path.join(dst_path, seq_id)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            img_path = os.path.join(save_path, "%05d.png" % frame_ind)
            canvas.save(img_path)
            print("{}/{} saved".format(seq_id, "%05d.png" % frame_ind))

        dst_first_path = os.path.join(dst_path, seq_id, "00000.png")
        shutil.copyfile(fframe_path, dst_first_path)
        print("copied from {} to {}".format(fframe_path, dst_first_path))