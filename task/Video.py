import zmq
import cv2
import numpy as np
from pathlib import Path
import os
import time

def publishVideo(config):
    ctx = zmq.Context()

    send_sock = ctx.socket(zmq.PUB)
    send_sock.bind("tcp://*:" + config['camera_port'])
 
    view_sock = ctx.socket(zmq.PUB)
    view_sock.bind("tcp://*:" + config['view_port'])

    time.sleep(1)
    
    count = 0
    frames = []

    for i in range(config['video_length']):
        #Center crop
        frame = cv2.imread(os.path.join(Path(__file__).parent.parent.resolve(),Path(f"{config['video_path']}{i+1:04d}.png")))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_frame = frame#[(config['video_height'] - config['crop_height']) // 2: (config['video_height'] + config['crop_height']) // 2,(config['video_width'] - config['crop_width']) // 2:(config['video_width'] + config['crop_width']) //2]
        if(config['use_view']): view_sock.send(crop_frame.tobytes())
        frames.append(crop_frame)
        count += 1
        if count == config['num_frames']:
            video = np.stack(frames, axis=0) # (T, H, W, C)    
            video = np.transpose(video, axes=[0, 3, 1, 2]) # (T,C,H,W)
            video = np.float32(video[np.newaxis, :]) # (B,T,C,H,W)
            video /= 255
            send_sock.send(video.tobytes())
            count =  15
            del frames[: config['num_frames'] - 15]
            if(config['verbose']): print(f"Send video with shape {video.shape} and type {video.dtype}")
        time.sleep(0.03)
    send_sock.close()
    view_sock.close()
    ctx.term()
    
if __name__ == "__main__":
    publishVideo()
