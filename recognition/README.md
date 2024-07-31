#  Obtaining the Sign Language Recognition
Our work heavily builds on [A Simple Multi-modality Transfer Learning Baseline for Sign Language Translation, CVPR2022](https://arxiv.org/abs/2203.04287).
We mainly follow their implementation for the sign recognition model and we refer to [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) for their complete code and installation requirements.

Running our model requires the following models:
- *s3ds_actioncls_ckpt* (from https://github.com/kylemin/S3D)
- *s3ds_glosscls_ckpt* (from https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork)
- *phoenix-2014t_s2g/best.ckpt* (from https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork)

We provide the models in our [Google Drive](https://drive.google.com/drive/folders/1fnXKm4Mr86ABNouOZ8bQWiseYv0D_fmc?usp=sharing)/recognition.

We do not evaluate on the whole Phoenix-2014T dataset but we provide one example of the dataset in our shared folder and thank the dataset creators for providing the dataset (https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).

This directory includes the necessary code for generating the onnx file used in our application.

## Usage
We demonstrate our work using the Phoenix-2014T dataset.
To generate the onnx files, run the following:
```
dataset=phoenix-2014t
python prediction_lean.py --config experiments/configs/SingleStream/${dataset}_s2g.yaml
```


## Acknowledgements
We thank Y. Chen et al. for their amazing work and for open-sourcing their code and trained models.