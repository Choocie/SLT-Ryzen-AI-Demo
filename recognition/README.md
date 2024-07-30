#  Obtaining the Sign Language Recognition
Our work heavily builds on [A Simple Multi-modality Transfer Learning Baseline for Sign Language Translation, CVPR2022](https://arxiv.org/abs/2203.04287).
We mainly follow their implementation for the sign recognition model and we refer to [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) for their complete code, the installation requirements, and information about how to acquire the pre-trained models.
This directory includes the necessary code for generating the onnx file used in our application.

## Usage
We demonstrate our work using the Phoenix-2014T dataset.
To generate the onnx files, run the following:
```
dataset=phoenix-2014t
python prediction_lean.py --config experiments/configs/SingleStream/${dataset}_s2t.yaml
```


## Acknowledgements
We thank Y. Chen et al. for their amazing work and for open-sourcing their code.