# Audio deepfake fraud detection system

A machine learning-based system for detecting fraudulent audio recordings using deep learning techniques. This project is a working proof-of-concept implementation of the solution described in the [Hyperplane article on Audio Deepfake Fraud Detection](https://thehyperplane.substack.com/p/audio-deepfake-fraud-detection-system).

![title](images/Audio%20deepfake%20fraud%20detection%20system.png)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

## Overview

This project implements a deep learning-based solution for detecting fraudulent audio recordings, addressing a critical challenge in today's digital landscape where voice-based fraud is becoming increasingly sophisticated. According to recent reports, impersonation scams involving cloned voices generate over $25 billion in losses annually, with the technology required to replicate voices being easily available online.

The system uses a combination of:
- Pre-trained ResNet18 model with Bi-GRU for classification
- Mel spectrograms with masking for robust feature extraction
- Advanced data augmentation techniques including time shift, noise addition, pitch shift, and gain adjustments
- A balanced dataset combining ASV Spoof 2019 and Orpheus-generated samples

The project achieves 95-97% precision and recall in detecting fraudulent audio, making it particularly valuable for industries such as:
- Finance
- Customer Support
- Telecommunications
- Healthcare

## Features

- High-accuracy audio fraud detection (95-97% precision & recall)
- Advanced deep learning model combining ResNet18 and Bi-GRU
- Robust feature extraction using Mel spectrograms with masking
- Comprehensive data augmentation pipeline:
  - Time shifting
  - Noise addition
  - Pitch shifting
  - Spectrogram masking
- User-friendly web interface built with Streamlit
- Docker support for easy deployment
- Python 3.11+ compatibility
- Efficient dependency management with uv

## Prerequisites

- Python 3.11 or higher
- uv
- Docker (optional, for containerized deployment)
- Make (for using Makefile commands)

## Installation

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/mlvanguards/fraud-audio-detection.git
cd fraud-audio-detection
```

2. Install dependencies:
```bash
uv sync
```

3. Activate a virtual environment:
```bash
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### Docker Setup

Build and run using Docker:
```bash
docker build -t fraud-audio-detection .
docker run -p 8501:8501 --rm fraud-audio-detection
```

## Usage

### Running Locally

Start the Streamlit application:
```bash
streamlit run src/main.py
```

The application will be available at `http://localhost:8501`

### Using Docker

The application will be available at `http://localhost:8501` after running the Docker container.

## Project Structure

```
fraud-audio-detection/
├── data/               # Data directory
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
├── tests/            # Test files
├── Dockerfile        # Docker configuration
├── Makefile         # Build automation
├── pyproject.toml   # Project configuration
└── README.md        # This file
```

## Development

### Key Dependencies

- PyTorch (2.7.0)
- TorchAudio (2.7.0)
- TorchVision (0.22.0)
- Streamlit (1.45.1)
- SoundFile (0.13.1)


## License

This project is licensed under the terms of the license included in the repository.