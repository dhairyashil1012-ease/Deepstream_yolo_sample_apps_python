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
git clone https://github.com/dhairyashil1012-ease/Deepstream_yolo_sample_apps_python.git

cd Deepstream_yolo_sample_apps_python
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

---

# Launch DeepStream Container

Mount the DeepStream sources directory inside the container.

```bash
docker run -it --rm \
--gpus all \
--network host \
-v /opt/nvidia/deepstream/deepstream-8.0/sources:/opt/nvidia/deepstream/deepstream-8.0/sources \
nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
```

---

# Navigate to Application Directory

Inside the container:

```bash
cd /opt/nvidia/deepstream/deepstream-8.0/sources/Deep_Stream_Apps_Yolo/DeepStream_Yolo_Apps/deepstream_python_apps/apps/Sample_test_rtsp
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

# Running the Application

Start the pipeline:

```bash
python3 sample_rtsp_in_out.py
```

If your application supports command-line arguments, refer to the source code.

---

# Viewing RTSP Output

Once the pipeline starts successfully, an RTSP server is created.

View the stream using FFplay:

```bash
ffplay rtsp://<SERVER_IP>:8554/<STREAM_NAME>
```

Example:

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

# Verifying RTSP Server

Check whether the RTSP server is running:

```bash
netstat -tulpn | grep 8554
```

Expected result:

```text
tcp        0      0 0.0.0.0:8554
```

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

## RTSP Stream Not Accessible

Verify:

```bash
netstat -tulpn | grep 8554
```

Check:

* Container started with `--network host`
* RTSP port is available
* Firewall is not blocking the port

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
