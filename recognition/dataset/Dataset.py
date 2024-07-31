from tkinter.filedialog import Open
import torch, pickle
import gzip, os
import numpy as np
from utils.misc import get_logger


class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split):
        super(SignLanguageDataset, self).__init__()
        self.split = split #train, val, test
        self.dataset_cfg = dataset_cfg
        self.load_annotations()
        self.input_streams = dataset_cfg.get('input_streams',['rgb'])
        self.logger = get_logger()
        self.name2keypoints = None

    def load_annotations(self):
        self.annotation_file = self.dataset_cfg[self.split]
        with gzip.open(self.annotation_file, 'rb') as f:
            self.annotation = pickle.load(f)
            self.annotation = self.annotation[:1]
            print('Selecting one annotation.')
        for a in self.annotation:
            a['sign_features'] = a.pop('sign',None)

        for feature_name in ['head_rgb_input','head_keypoint_input']:
            filename = self.dataset_cfg.get(self.split+f'_{feature_name}','')
            if os.path.isfile(filename):
                with gzip.open(filename, 'rb') as f:
                    annotation = pickle.load(f)
                name2feature = {a['name']:a['sign'] for a in annotation}
                for a in self.annotation:
                    a[feature_name] = name2feature[a['name']]

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        return {k:v for k, v in self.annotation[idx].items() \
            if k in [
                'name','gloss','text','num_frames','sign',
                'head_rgb_input','head_keypoint_input',
                'inputs_embeds_list']}

def build_dataset(dataset_cfg, split):
    dataset = SignLanguageDataset(dataset_cfg, split)
    return dataset