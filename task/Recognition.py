import signal
import sys
import errno
import os
from pathlib import Path

import numpy as np

import zmq
import cv2

import onnx
import onnxruntime as ort

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exit...')
    sys.exit(0)
    
def generateGlossDatabase(path):
    with open(path, 'r') as file:
        data = file.read().replace('\n', '').replace('\'', '')    
        glossList = data.split(',')
        return dict(zip(range(len(glossList)), glossList))

def recognizeGloss(config):
    
    gloss_database_name = "resources/glosses.txt"
    glosses = generateGlossDatabase(gloss_database_name)
    
    provider_options = [{
        'config_file': config['provider_config'],
        'cacheDir': str(Path(__file__).parent.resolve()),
        'cacheKey': 'modelcachekey_quick'
    }]

    session = ort.InferenceSession(config['model_name'],
        providers=[config['recognition_provider']],
        provider_options=provider_options)

    ctx = zmq.Context()

    rcv_sock = ctx.socket(zmq.SUB)
    rcv_sock.setsockopt(zmq.CONFLATE, 1)
    rcv_sock.setsockopt(zmq.RCVTIMEO, 5000)
    rcv_sock.connect("tcp://127.0.0.1:" + config['camera_port'])
    rcv_sock.subscribe("")
    
    gloss_sock = ctx.socket(zmq.PUB)
    gloss_sock.bind("tcp://*:" + config['gloss_port'])
    
    while True:
        try:
            serialized_video = rcv_sock.recv()
        except zmq.error.Again as e:
            print('No available video stream found. Aborting...')
            sys.exit(0)
        video = np.frombuffer(serialized_video, dtype=np.float32).reshape(1,config['num_frames'],3, config['crop_height'], config['crop_width'])
        if(config['verbose']): print(f"Received Frame Buffer with dimensions {video.shape}")
        #model_input = np.float32(video[np.newaxis, :])
        #model_input /= 255
        #model_input = np.flip(model_input,axis=2)
        
        #if(config['verbose']): print(f"Start inference with {model_input.shape}")
        outputs = session.run(['onnx::LogSoftmax_729'], {
            'sgn_videos': video,
            'onnx::Mul_8': np.zeros((1),dtype=np.float32)})
        
        ids = np.argmax(outputs[0][:,:,:],axis=2).flatten()
        filtered_ids = [i for i in ids if i > 3]
        gloss = [glosses[i] for i in filtered_ids]
        
        if(config['use_view'] and len(gloss) > 0):
            print(','.join(gloss))
            gloss_sock.send(','.join(gloss).encode('ascii'))
        
        #Small test
        #video = np.transpose(video, axes=[0, 2, 3, 1]) # (T,C,H,W) -> (T, H, W, C)   
        #for i in range(40):
        #    cv2.imwrite(f"test_{i:d}.png",video[i,:,:,:])
        #image = cv2.putText(video[0,:,:,:].copy(), ' '.join(gloss), (30,config['crop_height'] - 30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1, cv2.LINE_AA) 
        #cv2.imshow("Receiver stream", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    rcv_sock.close()
    gloss_sock.close()
    ctx.term()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    recognizeGloss()
    