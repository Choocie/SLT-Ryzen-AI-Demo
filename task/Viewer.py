import zmq
import cv2
import numpy as np
import sys


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exit...')
    sys.exit(0)
    
def subscribeViewer(config):
    
    text = 'Hello World!'
    
    ctx = zmq.Context()

    rcv_sock = ctx.socket(zmq.SUB)
    #rcv_sock.setsockopt(zmq.CONFLATE, 1)
    rcv_sock.setsockopt(zmq.RCVTIMEO, 5000)
    rcv_sock.connect("tcp://127.0.0.1:" + config['view_port'])
    rcv_sock.subscribe("")
    
    gloss_sock = ctx.socket(zmq.SUB)
    gloss_sock.connect("tcp://127.0.0.1:" + config['gloss_port'])
    gloss_sock.subscribe("")
    
    #writer = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (config['crop_width'],config['crop_height']))
    
    while True:
        
        try:
            serialized_frame = rcv_sock.recv()
        except zmq.error.Again as e:
            print('No available video stream found. Aborting...')
            sys.exit(0)
        
        try:
            text = gloss_sock.recv(flags=zmq.NOBLOCK).decode("utf-8")
        except zmq.error.Again as e:
            #Do Nothing
            pass

        frame = np.frombuffer(serialized_frame, dtype=np.uint8).reshape(config['crop_height'], config['crop_width'],3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, text, (20,config['crop_height'] - 30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1, cv2.LINE_AA) 
        cv2.imshow('Demo',frame)
        
        #writer.write(frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    #writer.release()
    rcv_sock.close()
    gloss_sock.close()
    ctx.term()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    subscribeViewer({
         'view_port'    : '1236',
         'crop_width'   : 210,
         'crop_height'  : 260,
         'verbose'      : True
    })
