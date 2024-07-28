import zmq
import cv2
import numpy as np

def publishCamera(config):
    ctx = zmq.Context()

    send_sock = ctx.socket(zmq.PUB)
    send_sock.bind("tcp://*:" + config['camera_port'])
    
    count = 0
    frames = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if ret:
            #Center crop
            crop_frame = frame[(config['camera_height'] - config['crop_height']) // 2: (config['camera_height'] + config['crop_height']) // 2, (config['camera_width'] - config['crop_width']) // 2:(config['camera_width'] + config['crop_width']) //2]
            frames.append(crop_frame)
            count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count == config['num_frames']:
            video = np.stack(frames, axis=0) # (T, H, W, C)    
            video = np.transpose(video, axes=[0, 3, 2, 1]) # (T,C,W,H)
            send_sock.send(video.tobytes())
            count =  config['num_frames'] // 2
            del frames[: config['num_frames'] // 2]
            if(config['verbose']): print(f"Send video with shape {video.shape} and type {video.dtype}")
    cap.release()
    cv2.destroyAllWindows()
    send_sock.close()
    ctx.term()
    
if __name__ == "__main__":
    publishVideo()
