# DeepStream Multi-Stream RTSP Pipeline (Python)

## Overview

This project is a DeepStream Python application built on top of NVIDIA DeepStream Python Apps.

The application supports:

* Multiple RTSP input streams
* DeepStream-based object detection
* Multi-object tracking
* On-Screen Display (OSD)
* RTSP output streaming
* GPU-accelerated inference using NVIDIA DeepStream

The application serves as a reference implementation for deploying real-time video analytics pipelines using DeepStream and Python.

---

# Features

* Multiple RTSP input streams
* DeepStream inference pipeline
* Object tracking
* RTSP output streaming
* GPU-accelerated processing
* Python-based implementation
* Easy configuration through DeepStream config files

---

# Prerequisites

Before running the application, ensure the following are installed on the host machine.

## Hardware Requirements

* NVIDIA GPU

## Software Requirements

* NVIDIA Driver
* Docker
* NVIDIA Container Toolkit

Verify GPU installation:

```bash
nvidia-smi
```

Expected output should display your GPU information.

---

# Clone Repository

Clone the repository on the host machine:

```bash

Make one Directory inside that 
cd directory and clone the repo.

git clone https://github.com/dhairyashil1012-ease/Deepstream_yolo_sample_apps_python.git


```

---

# DeepStream Docker Environment

This application has been tested using:

```text
nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
```

Pull the image:

```bash
docker pull nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
```

# Display Access:
 ```bash
 export DISPLAY=:0
 ```
 
 ## Close terminal and Open New Terminal 
```bash 
 go inside the main directory where you clone your repo. not inside the clone repo.
 cd main_directory 

 xhost +
```

---

# Launch DeepStream Container

Mount the DeepStream sources directory inside the container.

```bash
docker run -it  --network=host --gpus all -e DISPLAY=$DISPLAY --device /dev/snd -v /tmp/.X11-unix/:/tmp/.X11-unix -v $(pwd)/Deepstream_yolo_sample_apps_python:/opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_sample_app nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
```

---


## Come out side the container without stop

```bash
 Ctrl+P followed by Ctrl+Q
 ```

# Open vscode :
```bash
code .
```

# Attach Container Shell in vscode terminal


# Navigate to Application Directory

Inside the container:
you will see you main clone git repo. inside the sources 

```bash
cd /opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_sample_app
```

---

# Install Required Python Packages

## Install DeepStream Python Bindings

```bash
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl

pip3 install pyds-1.2.2-cp312-cp312-linux_x86_64.whl
```

Verify installation:

```bash
python3 -c "import pyds; print('pyds installed successfully')"
```

---

## Install CUDA Python

Inside the application directory:

```bash
pip3 install cuda-python
```

Verify installation:

```bash
python3 -c "import cuda; print('cuda-python installed successfully')"
```

---

# Project Structure

```text
Sample_test_rtsp/
│
|──common/
├── sample_rtsp_in_out.py
│
├── dstest3_pgie_config.txt
├── dstest2_tracker_config.txt
│
└── README.md
```

---

# Pipeline Architecture

The application creates the following DeepStream pipeline:

```text
RTSP Input(s)
      │
      ▼
nvstreammux
      │
      ▼
nvinfer
      │
      ▼
nvtracker
      │
      ▼
nvdsosd
      │
      ▼
Encoder
      │
      ▼
RTSP Server
      │
      ▼
RTSP Output
```

---

# Pipeline Components

## Source

Reads one or more RTSP streams.

## nvstreammux

Batches multiple input streams for GPU processing.

## nvinfer

Runs inference using TensorRT engine files.

## nvtracker

Tracks detected objects across consecutive frames.

## nvdsosd

Draws:

* Bounding boxes
* Labels
* Tracking IDs

## Encoder

Encodes processed frames.

## RTSP Server

Publishes processed video as an RTSP stream.

---

# Configuration Files

## dstest3_pgie_config.txt

Primary inference configuration.

Responsible for:

* Model loading
* Inference configuration
* Preprocessing settings
* Detection parameters

---

## dstest2_tracker_config.txt

Tracker configuration.

Responsible for:

* Tracker selection
* Tracker dimensions
* Performance tuning
* Tracking parameters

---

# Running the Application for only one input

Start the pipeline:

Before passing any 
```bash
To run:
  $ python3 deepstream_test_3.py -i <uri1> [uri2] ... [uriN] [--no-display] [--silent]
e.g.
  $ python3 deepstream_test_3.py -i file:///home/ubuntu/video1.mp4 file:///home/ubuntu/video2.mp4
  $ python3 deepstream_test_3.py -i rtsp://127.0.0.1/video1 rtsp://127.0.0.1/video2 -s
```


for example file path just for reference deepstream all sample files are present in streams directory :
``` file:///opt/nvidia/deepstream/deepstream-8.0/samples/streams/sample_720p.mp4  ```


```bash
python3 sample_rtsp_in_out.py -i file:///opt/nvidia/deepstream/deepstream-8.0/samples/streams/sample_720p.mp4
```

## If you encounter an error like :

```bash
ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:58 Cannot access ONNX file '/opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_sample_app/../../../../../../samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx'
ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:722 failed to build network since parsing model errors.
ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:563 failed to build network.
0:00:03.376858854   346     0x34020120 ERROR                nvinfer gstnvinfer.cpp:679:gst_nvinfer_logger:<primary-inference> NvDsInferContext[UID 1]: Error in NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:2143> [UID = 1]: build engine file failed
0:00:03.815471579   346     0x34020120 ERROR                nvinfer gstnvinfer.cpp:679:gst_nvinfer_logger:<primary-inference> NvDsInferContext[UID 1]: Error in NvDsInferContextImpl::generateBackendContext() <nvdsinfer_context_impl.cpp:2229> [UID = 1]: build backend context failed
```

```bash
Check file location in

ls /opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/

```
 
# you will get to see something like this files .
```bash
cal_trt.bin  labels.txt  resnet18_trafficcamnet_pruned.onnx
```
# So go and check config files path you just need to update the path of file .
```bash
onnx-file=../../../../../../samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx
model-engine-file=../../../../../..//samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine
```

This is the traveling path of model file .

../../ :- one '../' shows that one directory so our models present in deepstream-8.0/samples/models/Prrimary.../ in this directory but our code is present in deepstream-8.0/sources/deepstream_apps_python 
so one '../' represent to one directory . our common directory is deepstream-8.0 so you have to add '../' this in your config file till you not getting and deepstream-8.0 dir.  you have to look from your main code directory where your actual code is present . so dont confuse in this just update the path of model. 


``` bash
../../samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx
```

# Once you update the config file re-run the script by taking the reference of above command.

Once the pipeline starts successfully, an RTSP server is created on your ubuntu terminal .

---

# Viewing RTSP Output use your terminal of ubuntu . Go on ubuntu Terminal
```bash
sudo apt update
sudo apt install ffmpeg

```


Once the pipeline starts successfully, an RTSP server is created on your ubuntu terminal .

View the stream using FFplay:

```bash
ffplay rtsp://<SERVER_IP>:8554/<STREAM_NAME>
```

Example command to execute on terminal after above steps:

```bash
ffplay rtsp://localhost:8554/ds-test
```

Supported clients:

* FFplay
* VLC
* GStreamer
* OpenCV
* FFmpeg

---


# Troubleshooting

## pyds Module Not Found

Error:

```text
ModuleNotFoundError: No module named 'pyds'
```

Solution:

```bash
pip3 install pyds-1.2.2-cp312-cp312-linux_x86_64.whl
```

---

## cuda-python Module Not Found

Error:

```text
ModuleNotFoundError: No module named 'cuda'
```

Solution:

```bash
pip3 install cuda-python
```

---

## Engine File Not Found

Common errors:

```text
Failed to create NvDsInferContext

Failed to open model file

Could not find engine file

Cannot access model path
```

Verify:

* Engine file exists
* Model file exists
* Configuration paths are correct
* File permissions are correct

---

## Model Path Issues

The application relies on model and TensorRT engine files referenced inside the DeepStream configuration files.

Verify paths inside:

```text
dstest3_pgie_config.txt
```

Sample DeepStream models are typically available at:

```text
/opt/nvidia/deepstream/deepstream-8.0/samples/models/
```

List available models:

```bash
ls /opt/nvidia/deepstream/deepstream-8.0/samples/models/
```

Example detector models:

```bash
ls /opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/
```

Example contents:

```text
labels.txt
resnet18_trafficcamnet_pruned.onnx
resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine
```

If model files are located elsewhere, update the corresponding paths in the configuration files.

Restart the application after making changes.


---
# For Multiple Inputs 

```bash
python3 sample_rtsp_in_out.py -i file:///opt/nvidia/deepstream/deepstream-8.0/samples/streams/sample_720p.mp4 file:///opt/nvidia/deepstream/deepstream-8.0/samples/streams/sample_1080_h265.mp4

change in config file in model name b1 convert to b2 for 2 inputs 
OR for RTSP input understand what is rtsp and then try to execute because rtsp means real time streaming protocol 

# For example

python3 sample_rtsp_in_out.py -i rtsp://127.0.0.1/video1 rtsp://127.0.0.1/video2
```


---

## GPU Not Detected

Verify:

```bash
nvidia-smi
```

Ensure the container is launched with:

```bash
--gpus all
```

---

# Notes

* Tested with DeepStream 8.0
* Requires NVIDIA GPU
* Requires NVIDIA Container Toolkit
* Model paths may vary between systems
* Update configuration files if model locations change
* TensorRT engine files should match the target GPU architecture

---

# Author

**Dhairyashil Narute**

DeepStream Multi-Stream RTSP Pipeline (Python)
