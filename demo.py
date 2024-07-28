import zmq
import time
import sys
from  multiprocessing import Process

from task.Camera import publishCamera
from task.Recognition import recognizeGloss
from task.Video import publishVideo


config = {
     'model_name' :         "resources/singlestream.onnx",
     'driver_path':         'voe-4.0-win_amd64',
     'recognition_provider':'VitisAIExecutionProvider',
     'provider_config' :    '../voe-4.0-win_amd64/vaip_config.json',
     'num_frames':           40,
     'crop_width':           210,
     'crop_height':          260,
     'camera_width':         640,
     'camera_height':        480,
     'video_width':          210,
     'video_height':         260,
     'video_path':           "resources/video/images",
     'video_length':         40,
     'camera_port':          '1235',
     'verbose':              True,
     'use_camera':           False
    }

if __name__ == "__main__":
    if(config['use_camera']):
        Process(target=publishCamera, args=(config,)).start()
    else:
        Process(target=publishVideo, args=(config,)).start()
    Process(target=recognizeGloss, args=(config,)).start()