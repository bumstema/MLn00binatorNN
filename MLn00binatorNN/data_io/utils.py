import json
import cv2
import torchvision.transforms as transforms
import sys, os, os.path
import numpy as np
import torch
import PIL
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from .const import NN_TRAINING_PATH

#------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#------------------------------------------------------------
def save_json_file(data, filename):
    serialized_data = data
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, default=lambda o: o.__dict__, ensure_ascii=False)

#------------------------------------------------------------
def load_json_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data

#------------------------------------------------------------
def process_raw_frame(np_rgb):
    pic = datapoints.Image(PIL.Image.fromarray(np_rgb.astype(np.uint8)))
    return(pic)

#------------------------------------------------------------
def rgb2gray(rgb):
    return cv2.cvtColor( rgb, cv2.COLOR_RGB2GRAY)
    
#------------------------------------------------------------
def np_grayscale(rgb_img):
    return np.dot(rgb_img[...,:3], [0.2989, 0.5870, 0.1140])
  
#------------------------------------------------------------
def image_to_tensor(np_gray_img):
    pic = datapoints.Image(PIL.Image.fromarray((np_gray_img).astype(np.uint8)))
    pic = pic.type(dtype=torch.float32)
    pic = pic/255
    return pic
    
#------------------------------------------------------------
def tensor_to_pil_image( pic ):
    return transforms.ToPILImage()(pic)
    
#------------------------------------------------------------
def unpack_nested_list(listed_lists):
    return [val for sublist in listed_lists for val in sublist]
    
#------------------------------------------------------------



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------------------------------------------------------------
def PlotGraph( param_dict, filename='graph.png' ):

    df = pd.DataFrame(param_dict)
    x = df['episode']
    y = df['total_reward']

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y,'o-', lw=2)
    ax.fill_between(x, 0, y, alpha=0.2)

    majorLocator   = MultipleLocator(1)

    plt.xlabel(f'episode', fontsize=18)
    plt.ylabel(f'reward', fontsize=18)
    plt.title(f'level-1', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig( NN_TRAINING_PATH + f'{filename}')
    plt.close()


#------------------------------------------------------------
#------------------------------------------------------------
def load_mp4_as_frames( file_path ):

    vidcap = cv2.VideoCapture( file_path )

    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_frames = vidcap.get(cv2.CAP_PROP_FPS)
    ms_per_frame = (1/fps_frames)*1000
    print( f"Total Frames: {n_frames}  Frames/sec: {fps_frames}  ms/Frame: {ms_per_frame}")
    
    gif_frames = []
    dif_frames = []
    success,image = vidcap.read()
    print(f"{image.shape = }" )
    (h, w, c) = image.shape
    reshape_h = int(h)
    reshape_w = int(h * WIDTH / HEIGHT)
    print(f"{reshape_h = }  {reshape_w = }")
    frame = cv2.cvtColor(cv2.resize(image[:reshape_h, :reshape_w, :], DISPLAY, interpolation = cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
    idx = 0
    n_video_frames = 0
    n_changed_frames  = 0
    start_time = datetime.now()
    while success:
        n_video_frames += 1
        prev_frame = frame
        success, frame = vidcap.read()
        frame = frame[:reshape_h, :reshape_w, :]
        frame = cv2.resize(frame, DISPLAY, interpolation = cv2.INTER_AREA)
        frame = cv2.cvtColor(np.array(frame) , cv2.COLOR_BGR2RGB)
        dif_frame, dif_val  = framestep_image_difference(frame, prev_frame)
        if dif_val == 0: continue
        n_changed_frames += 1
        gif_frames.append(frame)
        dif_frames.append(dif_frame)
        text = detect_text_in_frame(frame)
        print(f"{n_video_frames = }\t{n_changed_frames = }\t{text = }")
        
        idx += 1
        if idx > 1000:
            save_gif_frames([Image.fromarray(pic).convert("RGB") for pic in gif_frames], ms_per_frame=ms_per_frame, file_name=f'training_frames.gif')
            save_gif_frames(dif_frames, ms_per_frame=ms_per_frame, file_name=f'training_frames_dif.gif')

            print(f"time: {(datetime.now()-start_time).seconds}")
            exit()
        
    gif_frames = (gif_frames[:n_frames:3])
    ms_per_frame = ms_per_frame * 3
    
    return gif_frames, ms_per_frame
    
#------------------------------------------------------------
def save_gif_frames(gif_frames, ms_per_frame=16.67, file_name=f'file_name.gif'):
    print(f"Now saving gif...")
    gif_frames[0].save(f'{file_name}', format='GIF', append_images=gif_frames[1:], save_all=True, loop=0, duration= ms_per_frame, disposal=0, optimize=True)
    
#------------------------------------------------------------
def progress(count, total, prefix='', suffix=''):
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'%s [%s] %s%s | %s/%s ... %s\r' % (prefix, bar, percents, '%', count, total, suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben
    if count == total: print(f"\n")
