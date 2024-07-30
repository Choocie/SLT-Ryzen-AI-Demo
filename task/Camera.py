import zmq
import cv2
import numpy as np

def publishCamera(config):
    ctx = zmq.Context()

    send_sock = ctx.socket(zmq.PUB)
    send_sock.bind("tcp://*:" + config['camera_port'])
    
    view_sock = ctx.socket(zmq.PUB)
    view_sock.bind("tcp://*:" + config['view_port'])
    
    count = 0
    frames = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if ret:
            #Center crop
            crop_frame = frame[(config['camera_height'] - config['crop_height']) // 2: (config['camera_height'] + config['crop_height']) // 2, (config['camera_width'] - config['crop_width']) // 2:(config['camera_width'] + config['crop_width']) //2]
            crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            if(config['use_view']): view_sock.send(crop_frame.tobytes())
            frames.append(crop_frame)
            count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count == config['num_frames']:
            video = np.stack(frames, axis=0) # (T, H, W, C)    
            video = np.transpose(video, axes=[0, 3, 1, 2]) # (T,C,H,W)
            video = np.float32(video[np.newaxis, :]) # (B,T,C,H,W)
            video /= 255 # Scale to 0 ... 1
            send_sock.send(video.tobytes())
            count =  15
            del frames[: config['num_frames'] - 15]
            if(config['verbose']): print(f"Send video with shape {video.shape} and type {video.dtype}")
    cap.release()
    view_sock.close()
    send_sock.close()
    ctx.term()
    
if __name__ == "__main__":
    publishCamera({
         'camera_port'  : '1235',
         'view_port'    : '1236',
         'num_frames'   : 40,
         'crop_width'   : 210,
         'crop_height'  : 260,
         'camera_width' : 640,
         'camera_height': 480,
         'verbose'      : True
    })
