import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


def generate_guide(patch, distribution, radius, line_width=2):
    height, width, _ = patch.shape
    center = (int(width / 2), height)
    for i in range(5):
        r = int(radius / 5 * (5-i))
        cv2.ellipse(patch, center, (r+line_width, r+line_width), 0, -180, 0, (255, 255, 255), -1)
        for j in range(5):
            cv2.ellipse(patch, center, (r, r), 0, -180+36*j, -180+36*j+36, (0, distribution[j*5+4-i]*255, 0), -1)
    for i in range(4):
        angle = math.pi / 5 * (4-i)
        endpoint = (int(center[0] + math.cos(angle) * radius), int(center[1] - math.sin(angle) * radius))
        cv2.line(patch, endpoint, center, (255, 255, 255), line_width)
    return patch



def draw_guide(img, distribution=np.arange(25)/25, radius=320, line_width=4):
    # img = cv2.imread(fname).astype(np.float64)
    height, width, _ = img.shape
    center = (int(width / 2), height)
    patch = img[height-radius-line_width:height, int(width/2-radius-line_width):int(width/2+radius+line_width), :].copy()
    patch = generate_guide(patch, distribution, radius, line_width)
    img[height-radius-line_width:height, int(width/2-radius-line_width):int(width/2+radius+line_width), :] = img[height-radius-line_width:height, int(width/2-radius-line_width):int(width/2+radius+line_width), :] * 0.5 + patch * 0.5
    return img


def draw_bbox(img, bbox, color=(0,0,255), tck=1):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=tck)
    return img

def draw_bbox_with_info(img, bbox, text, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale=1, bg_color=(0,128,128), text_color=(0,0,255), tck=1):
    # by default the text will be put at the top-left corner of bbox
    # bbox: list of [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=tck)[0]
    box_coords = ((x1, y1), (x1+text_width+2, y1-text_height-2))
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(img, text, (x1, y1), font, font_scale, text_color)
    img = draw_bbox(img, bbox, text_color, tck)
    return img
    
if __name__ == '__main__':
    draw_guide()
