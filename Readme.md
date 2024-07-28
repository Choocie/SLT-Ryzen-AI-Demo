# Sign Language Translation for AMD Ryzen AI 

This software is developed by Olaf Duenkel and Marc Neu. 

### Hardware Requirements

- AMD Ryzen AI compatible processor
- USB Camera (for on-air demonstration)
- Keyboard, Mouse, PnP Monitor

### Prerequisites

 - AMD Ryzen AI Software Version 1.1
 - Python 3.9

### Installation

1. Setup a new virtual environment: `pip install -r requirements.txt`

2. You may have to make adjustments to the config dictionary in `demo.py`. Please make sure that the driver binaries are linked correctly and your model files are referenced accordingly.
 
### Running

Currently, there are two operation modes supported. In both cases, start the application by executing `python ./demo.py` in the project root directory.

1. For testing, we support loading an image sequence. To use this mode, set `use_camera` to `False`.

2. For demonstrating the system "on-air", set  `use_camera` to `True`. The application expects a connected USB Camera on your computer.
