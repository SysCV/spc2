import os
from shutil import copyfile
import PIL.Image as Image
import argparse

parser = argparse.ArgumentParser(description='import args')
parser.add_argument('--src', type=str, help="src dir storing the produced recordings")
parser.add_argument('--dst', type=str, help="dst dir to store merged demo images and videos")
parser.add_argument('--seg_w', type=int, default=100, help="value of segmentation width")
parser.add_argument('--seg_h', type=int, default=100, help="value of segmentation height")
parser.add_argument('--obs_w', type=int, default=500, help="value of observation width")
parser.add_argument('--obs_h', type=int, default=500, help="value of observation height")
args = parser.parse_args()

seg_col, seg_row = 2,5
seg_w = args.seg_w
seg_h = args.seg_h
obs_w = args.obs_w
obs_h = args.obs_h

def produce(demo_dir, demo_dst_dir):
    demo_dirs = os.listdir(demo_dir)
    for img_dir in demo_dirs:
        img_path = os.path.join(demo_dir, img_dir, "obs_bbox.png")
        dst_path = os.path.join(demo_dst_dir, "img", "{}.png".format(img_dir.zfill(4)))
        copyfile(img_path, dst_path)
        seg_path = os.path.join(demo_dir, img_dir, "outcome")
        seg_overall = Image.new('RGB', (obs_w + seg_col * seg_w, seg_row * seg_h))
        for i in range(10):
            seg_name = "seg{}_bbox.png".format(i+1)
            seg_img_name = os.path.join(seg_path, seg_name)
            from_img = Image.open(seg_img_name)
            # from_img = from_img.convert('RGB')
            from_img = from_img.resize((seg_w, seg_h), Image.ANTIALIAS)
            col_index = int(i % seg_col)
            row_index = int((i/seg_col) % seg_row)
            seg_overall.paste(from_img, (obs_w + col_index * seg_w, row_index * seg_h))
        obs_img = Image.open(img_path)
        obs_img = obs_img.resize((obs_w, obs_h), Image.ANTIALIAS)
        seg_overall.paste(obs_img, (0,0))
        seg_overall.save(os.path.join(demo_dst_dir, "combined", "{}.png".format(img_dir.zfill(4))))

def merge(demo_dst_dir):
    img_target_dir = os.path.join(demo_dst_dir, "combined")
    combine_video_cmd = "ffmpeg -f image2 -i {}/%04d.png  -vcodec libx264 -r 15 {}/demo_out.mp4".format(img_target_dir, demo_dst_dir)
    os.system(combine_video_cmd)

def main():
    demo_dir = args.src
    demo_dst_dir = args.dst
    if not os.path.exists(demo_dst_dir):
        os.makedirs(demo_dst_dir)
        os.makedirs(os.path.join(demo_dst_dir, "img"))
        os.makedirs(os.path.join(demo_dst_dir, "combined"))

    produce(demo_dir, demo_dst_dir)
    merge(demo_dst_dir)
    

if __name__ == "__main__":
    main()


