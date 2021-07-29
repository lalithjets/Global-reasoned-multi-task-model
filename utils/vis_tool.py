import time

import random
import numpy as np
import matplotlib
import torch as t

matplotlib.use('Agg')
from matplotlib import pyplot as plot
from PIL import Image, ImageDraw, ImageFont


def vis_img(img, node_classes, bboxs,  det_action, data_const, score_thresh = 0.7):
    
    Drawer = ImageDraw.Draw(img)
    line_width = 3
    outline = '#FF0000'
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=25)
    
    im_w,im_h = img.size
    node_num = len(node_classes)
    edge_num = len(det_action)
    tissue_num = len(np.where(node_classes == 1)[0])
    
    for node in range(node_num):
        
        r_color = random.choice(np.arange(256))
        g_color = random.choice(np.arange(256))
        b_color = random.choice(np.arange(256))
        
        text = data_const.instrument_classes[node_classes[node]]
        h, w = font.getsize(text)
        Drawer.rectangle(list(bboxs[node]), outline=outline, width=line_width)
        Drawer.text(xy=(bboxs[node][0], bboxs[node][1]-w-1), text=text, font=font, fill=(r_color,g_color,b_color))
  
    edge_idx = 0
    
    for tissue in range(tissue_num):
        for instrument in range(tissue+1, node_num):
            
            #action_idx = np.where(det_action[edge_idx] > score_thresh)[0]
            action_idx = np.argmax(det_action[edge_idx])
#             print('det_action', det_action[edge_idx])
#             print('action_idx',action_idx)
            
            text = data_const.action_classes[action_idx]
            r_color = random.choice(np.arange(256))
            g_color = random.choice(np.arange(256))
            b_color = random.choice(np.arange(256))
        
            x1,y1,x2,y2 = bboxs[tissue]
            x1_,y1_,x2_,y2_ = bboxs[instrument]
            
            c0 = int(0.5*x1)+int(0.5*x2)
            c0 = max(0,min(c0,im_w-1))
            r0 = int(0.5*y1)+int(0.5*y2)
            r0 = max(0,min(r0,im_h-1))
            c1 = int(0.5*x1_)+int(0.5*x2_)
            c1 = max(0,min(c1,im_w-1))
            r1 = int(0.5*y1_)+int(0.5*y2_)
            r1 = max(0,min(r1,im_h-1))
            Drawer.line(((c0,r0),(c1,r1)), fill=(r_color,g_color,b_color), width=3)
            Drawer.text(xy=(c1, r1), text=text, font=font, fill=(r_color,g_color,b_color))

            edge_idx +=1

    return img