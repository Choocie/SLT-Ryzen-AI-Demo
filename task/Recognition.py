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
    
def ctc_beam_search_decoder(logits, sequence_lengths, beam_width=10, blank=0):
    """
    Perform CTC beam search decoding.

    Args:
        logits: A tensor of shape [batch_size, max_time, num_classes] containing the output logits.
        sequence_lengths: A tensor of shape [batch_size] containing the sequence lengths.
        beam_width: The width of the beam search.
        blank: The index of the blank label.

    Returns:
        A list of decoded sequences (one per batch).
    """
    batch_size, max_time, num_classes = logits.shape
    log_probs = np.log(np.exp(logits)/np.sum(np.exp(logits),axis=2,keepdims=True))

    
    results = []
    for b in range(batch_size):
        beams = [([], 0)]  # Initialize with an empty path and zero log-probability
        
        for t in range(sequence_lengths[b]):
            new_beams = []
            for prefix, score in beams:
                for c in range(num_classes):
                    new_prefix = prefix + [c]
                    new_score = score + log_probs[b, t, c].item()
                    new_beams.append((new_prefix, new_score))
            
            # Sort by score and keep the top beam_width beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams
        
        # Choose the best path (excluding the blank label)
        best_path = beams[0][0]
        decoded = []
        prev_char = None
        for char in best_path:
            if char != blank and char != prev_char:
                decoded.append(char)
            prev_char = char
        results.append(decoded)
    
    return results


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
        
        decoded_sequences = ctc_beam_search_decoder(outputs[0], np.array([10,]), beam_width=5, blank=0)
        
        #ids = np.argmax(outputs[0][:,:,:],axis=2).flatten()
        #filtered_ids = [i for i in ids if i > 3]
        #gloss = [glosses[i] for i in filtered_ids]
        gloss = [glosses[i] for i in decoded_sequences[0]]
        print(','.join(gloss))

        if(config['use_view'] and len(gloss) > 0):
            gloss_sock.send(','.join(gloss).encode('ascii'))
     
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    rcv_sock.close()
    gloss_sock.close()
    ctx.term()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    recognizeGloss()
    