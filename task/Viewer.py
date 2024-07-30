import zmq
import cv2
import numpy as np
import sys
import time

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exit...')
    sys.exit(0)
    
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, 1, 1)
    text_w, text_h = text_size
    cv2.rectangle(img, (0,y), (220, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h - 1), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (255,255,255), 1, cv2.LINE_AA) 
    
def update_text(text):
    words = text.split()
    new_list = []
    while( len(words) > 0):
        buffer = ' '.join(words[:5])
        del words[:5]
        new_list.append(buffer)
    return new_list
    
def subscribeViewer(config):
    
    gloss = ' '
    text = []
    
    ctx = zmq.Context()

    rcv_sock = ctx.socket(zmq.SUB)
    #rcv_sock.setsockopt(zmq.CONFLATE, 1)
    rcv_sock.setsockopt(zmq.RCVTIMEO, 5000)
    rcv_sock.connect("tcp://127.0.0.1:" + config['view_port'])
    rcv_sock.subscribe("")        

    gloss_sock = ctx.socket(zmq.SUB)
    gloss_sock.connect("tcp://127.0.0.1:" + config['gloss_port'])
    gloss_sock.subscribe("")
    
    text_sock = ctx.socket(zmq.SUB)
    text_sock.connect("tcp://127.0.0.1:" + config['translate_port'])
    text_sock.subscribe("")
    
    writer = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (config['crop_width'],config['crop_height']))

    while True:
        
        try:
            serialized_frame = rcv_sock.recv()
        except zmq.error.Again as e:
            print('No available video stream found. Aborting...')
            sys.exit(0)
        
        try:
            gloss = gloss_sock.recv(flags=zmq.NOBLOCK).decode("utf-8")
        except zmq.error.Again as e:
            #Do Nothing
            pass
        
        try:
            response = text_sock.recv(flags=zmq.NOBLOCK).decode("utf-8")
            response = response.splitlines()[2]
            print(response)
            text = update_text(response)
            gloss = ' '
        except zmq.error.Again as e:
            #Do Nothing
            pass

        frame = np.frombuffer(serialized_frame, dtype=np.uint8).reshape(config['crop_height'], config['crop_width'],3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        draw_text(frame, gloss, pos=(10,config['crop_height'] - 40))
        
        for i,line in enumerate(text):
            draw_text(frame, line, pos=(10,40 + 20*i))

        cv2.imshow('Test',frame)
        
        writer.write(frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()
    rcv_sock.close()
    gloss_sock.close()
    text_sock.close()
    ctx.term()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    subscribeViewer({
         'view_port'    : '1236',
         'crop_width'   : 210,
         'crop_height'  : 260,
         'verbose'      : True
    })
