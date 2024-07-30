import zmq
import time
import sys
from  multiprocessing import Process

from task.Camera import publishCamera
from task.Recognition import recognizeGloss
from task.Translate import publishTranslate
from task.Video import publishVideo
from task.Viewer import subscribeViewer


config = {
     'model_name' :         "resources/singlestream_40.onnx",
     'gloss_database_name' : "resources/glosses_en.txt",
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
     'video_length':         1800,
     'camera_port':          '1235',
     'view_port':            '1236',
     'gloss_prompt_port':    '1237',
     'translate_port':       '1238',
     'gloss_port':           '1239',
     'verbose':              False,
     'use_camera':           True,
     'use_view':             True,
     'use_translation':      False,
    }

if __name__ == "__main__":
    if(config['use_camera']):
        Process(target=publishCamera, args=(config,)).start()
    else:
        Process(target=publishVideo, args=(config,)).start()
    Process(target=recognizeGloss, args=(config,)).start()
    if(config['use_view']):
        Process(target=subscribeViewer, args=(config,)).start()
