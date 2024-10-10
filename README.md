# FRC Robot Tracking and Analysis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/qualk/frc-robot-tracking/issues)
[![GitHub issues](https://img.shields.io/github/issues/qualk/frc-robot-tracking.svg)](https://github.com/qualk/frc-robot-tracking/issues)
[![GitHub stars](https://img.shields.io/github/stars/qualk/frc-robot-tracking.svg)](https://github.com/qualk/frc-robot-tracking/stargazers)

## Overview

This project aims to track and analyse robots in FIRST Robotics Competition (FRC) matches using computer vision techniques. It provides tools for object detection, tracking, and visualisation, including heatmap generation.

## Features

- Robot detection and tracking in FRC match videos
- Heatmap generation to visualise robot movement patterns
- Velocity and acceleration measurement (WIP, the velocity and acceleration measurements in `bytrack.py` are currently inaccurate due to noise problems.)
- Integration with FRC API and The Blue Alliance API for scoring data analysis (planned)
- Integration to write and read csv files for every instance of detections (planned)

## Requirements

- [Python 3.11.9](https://www.python.org/downloads/release/python-3119/ "This is the ninth (and last) bugfix release of Python 3.11"), [due to `inference`](https://github.com/roboflow/inference?tab=readme-ov-file#-install "A fast, easy-to-use, production-ready inference server for computer vision supporting deployment of many popular model architectures and fine-tuned models.").

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/qualk/frc-robot-tracking.git
    cd frc-robot-tracking
    ```

2. Set up a virtual environment (recommended):

    Using [`uv`](https://docs.astral.sh/uv/ "An extremely fast Python package and project manager, written in Rust.") (requires [separate installation](https://docs.astral.sh/uv/getting-started/installation/ "Install uv with our standalone installers or your package manager of choice.")):

    ```bash
    uv venv --python 3.11
    ```

   Then activate the environment (regardless of the option).
   
   If you are using Windows, use `venv\Scripts\activate`. <br>
   If using Linux or MacOS, use `source venv/bin/activate`. This might differ based on your shell.

3. Install the required packages:

    ```bash
    uv pip install -r requirements.txt
    ```

4. Copy the `.env.example` file to `.env` and fill in your API keys.

## Usage

### Robot Tracking

To track robots in a video:
```bash
python track.py
```

This script uses box annotator, trace annotator, and label annotator to mark detected robots in the video.

### Heatmap Generation

To generate a heatmap of robot detections:

```bash
python heatmap.py
```

This script analyses the file "video.mp4" and produces a heatmap visualisation of robot positions throughout the match.

### Velocity and Acceleration Measurement

Note: This feature is currently under development and **does not** produce accurate results due to noise issues.

```bash
python bytrack.py
```

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Roboflow for hosting the object detection model
- FIRST Robotics Competition for providing the data and inspiration
- The Blue Alliance for additional FRC data and API access
