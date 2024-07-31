import torch
import torch.nn.functional as F
from modelling.recognition import RecognitionNetwork
from utils.misc import get_logger

class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
        if self.task=='S2G':
            self.text_tokenizer = None
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type = 'video',
                transform_cfg=cfg['data']['transform_cfg'],
                input_streams = cfg['data'].get('input_streams','rgb'))
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

            if self.recognition_network.visual_backbone!=None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone.get_frozen_layers())
            if self.recognition_network.visual_backbone_keypoint!=None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone_keypoint.get_frozen_layers())
            


    def forward(self, is_train, translation_inputs={}, recognition_inputs={}, **kwargs):
        if self.task=='S2G':
            model_outputs = self.recognition_network(is_train=is_train, **recognition_inputs)
        else:
            raise NotImplementedError(f"Task {self.task} not implemented")
        return model_outputs

    
    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):          
        model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)  
        return model_outputs
    

    def ctc_beam_search_decoder(self,logits, sequence_lengths, beam_width=10, blank=0):
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
        batch_size, max_time, num_classes = logits.size()
        log_probs = F.log_softmax(logits, dim=2)  # Convert logits to log probabilities
        
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
                
                # Sort by score and keep the top `beam_width` beams
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

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        ctc_decode_output = self.ctc_beam_search_decoder (
                            gloss_logits, 
                            input_lengths,
                            beam_width=beam_size
                        )
        return ctc_decode_output

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()

def build_model(cfg):
    model = SignLanguageModel(cfg)
    return model.to(cfg['device'])