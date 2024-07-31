# Sign Language Translation for AMD Ryzen AI 

This software is developed by Olaf Duenkel and Marc Neu as part of the AMD Pervasive AI Challenge. 

### Hardware Requirements

- AMD Ryzen AI compatible processor
- USB Camera (for on-air demonstration)
- Keyboard, Mouse, PnP Monitor

### Prerequisites

 - AMD Ryzen AI Software Version 1.1
 - Python 3.9
 - LLama2 is required for the full translation model [Link](https://github.com/amd/RyzenAI-SW)

### Minimal Setup

For a minimal setup, e.g. without translation model, you need the following steps to set upt this repository:


1. Perform all installation steps from AMD Ryzen AI Software Version 1.1.
2. Clone the created conda environment using `conda create --name demo --clone <source_env>`.
3. Activate your enviroment using `conda activate demo`.
4. Install ZeroMQ and OpenCV using `pip install zmq opencv-python`.

You are now all set loadup the demo by calling `python demo.py` in the root directory. Before starting up the demo, make sure that all driver paths are linked correctly in the configuration dictionary inside the `demo.py`.
The onnx file can be acquired from [Google Drive](https://drive.google.com/drive/folders/1fnXKm4Mr86ABNouOZ8bQWiseYv0D_fmc?usp=sharing)/recognition/`singlestream_40.onnx`.

### Translation Setup

Using the LLama2 model, we are additionally supporting a full language translation for recognized glosses. There are some additional steps required for the setup:

1. Perform all installation steps for the LLama2 PyTorch model inside the RyzenAI-SW 1.1 repository. At the end you will have a conda environement called `ryzenai-transformers`.
2. Copy `task/Translate.py` from this repository into `RyzenAI-SW/example/transformers/models/llama2`.
3. Enable the environment via `conda activate ryzenai-transformers` and install ZermMQ into this environment through `pip install zmq`.
4. Run the Translation model via `python Translate.py` inside `RyzenAI-SW/example/transformers/models/llama2`.
5. Set the configuration entry `use_translation` inside `demo.py` to `True`.

You are now all set loadup the demo by calling `python demo.py` in the root directory.

### Running

Currently, there are two operation modes supported. In both cases, start the application by executing `python ./demo.py` in the project root directory.

1. For testing, we support loading an image sequence. To use this mode, set `use_camera` to `False`.

2. For demonstrating the system "on-air", set  `use_camera` to `True`. The application expects a connected USB Camera on your computer.


### Generation of onnx File
We refer to `recognition/README.md` for more details about how to acquire the onnx file.
