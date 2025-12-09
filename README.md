# Image Selection Tool

This project is a tool designed to select specific images based on the objects present within those images from a large dataset. The tool leverages advanced object detection models to identify and filter images containing specified objects.

## Features

- **Object Detection**: Utilizes [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) to identify objects within images.
- **Image Filtering**: Filters and selects images from a dataset based on the presence of specified objects.

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:lesani-ali/Image-Selection-Tool.git
    cd Image-Selection-Tool
    ```

2. Clone [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO):
    ```bash
    git clone git@github.com:IDEA-Research/GroundingDINO.git
    ```

2. Create environment:
    ```bash
    conda create --name image_selection python=3.8
    conda activate image_selection
    ```

3. Install the required dependencies:
    ```bash
    bash ./scripts/install_packages.sh
    ```
    - If you are unable to install packages using above script, follow these steps:
        1. Install basic packages
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install opencv-python matplotlib Pillow augly tensorboardX
        ```
        2. Install Grounding Dino:
        ```bash
        cd ./GroundingDINO
        pip install -e . 
        ```
        3. Download pre-trained model:
        ```bash
        mkdir -p weights
        cd weights
        wget -nc -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
        ```

## Usage

1. Define your dataset class. You can see example definition [here](datasets/MyData.py).


2. Run the tool:
    ```bash
    python main.py --input-dir "/path/to/your/dataset" --output-dir "/output/path" --dataset "SimSIN" --data-filename "all_large_release2.txt" --objects "all windows"
    ```
    - `--dataset` is name of dataset class that you defined for your own data.
    - `--data-filename` is text file including path to images in your dataset.
    - `--objects` is name of objects that you want to filter images based on that.
