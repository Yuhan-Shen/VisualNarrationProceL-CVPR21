#import torch as th
#from torch.utils.data import Dataset
import os
import numpy as np
import ffmpeg


def get_video_dim(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return height, width

def get_raw_video(video_path, framerate=10, n_frames=32):
    if os.path.isfile(video_path):
        #h, w = get_video_dim(video_path)
        #print(h, w)
        #height, width = h, w # TO DO
        height, width = 224, 224
        #framerate = 10
        #centercrop = True # To check

        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=framerate)
            .filter('scale', width, height)
        )
        #if centercrop:
        #    x = int((width - size) / 2.0)
        #    y = int((height - size) / 2.0)
        #    cmd = cmd.crop(x, y, size, size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
       

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        t = video.shape[0]
        rmd = t % n_frames
        #print(video.shape)
        if not rmd == 0:
            video = np.concatenate([video, np.zeros([n_frames-rmd, height, width, 3], dtype=np.uint8)])
        #print(video.shape)

        video = video.reshape([-1, n_frames, height, width, 3])
        video = video.astype('float32')
        video = (video) / 255
       
    return video

def get_text(text_path, pos=0):
    text = []
    with open(text_path, 'r') as f:
        for line in f.readlines():
            phrase = line.strip().split('\t')[pos]
            text.append(phrase)

    return text

def get_all_text(text_dir, idx, pos=0):
    text_path = sorted(os.listdir(text_dir))[idx]
    text = get_text(os.path.join(text_dir, text_path), pos)
    return text

