# U-Net Microbeads Detection

A deep learning project for detecting and segmenting microbeads using U-Net architecture with support for instance segmentation. This project handles both microscopy and smartphone camera images for bead detection and segmentation.

## Project Overview

This project implements a three-class segmentation approach for instance segmentation of microbeads:
- Background (Class 0)
- Bead Interior (Class 1)
- Bead Boundaries (Class 2)

The model uses an enhanced U-Net architecture with improved preprocessing for better cluster separation and watershed algorithm for instance segmentation.

## Project Structure

```
unet_microbeads/
├── .git/                            # Git repository data
├── README.md                        # This file
├── coco_selected_microscopy_training/   # Training data for microscopy images
├── coco_selected_phone_training/    # Training data for smartphone images
├── dataset/                         # Main dataset directory
│   ├── images/                      # Original images
│   ├── masks/                       # Binary masks
│   ├── circular_masks/              # Circular fitted masks
│   ├── instance_masks/              # Instance segmentation masks
│   ├── boundary_masks/              # Boundary masks
│   ├── three_class_masks/           # Three-class masks
│   ├── results/                     # Results from model predictions
│   ├── models/                      # Saved model weights
│   └── __pycache__/                 # Python cache files
├── environment_unet.yml             # Conda environment file
├── annotations.json                 # COCO format annotations
├── coco_combined.py                 # Script to combine multiple COCO annotations
├── generate_masks.py                # Script to generate masks from annotations
├── instance_segmentation_v5.py      # Main training script
├── prediction_v4.py                 # Prediction script for new images
└── new_images/                      # Directory for new images to predict
```

## Prerequisites

- Python 3.9
- Conda or Miniconda
- Docker (for CVAT labeling)
- CUDA-compatible GPU (optional, but recommended)
- Mac M2 support included (CPU mode)

## Setup Instructions

### 1. Environment Setup

Create the conda environment using the provided YAML file:

```bash
conda env create -f environment_unet.yml
conda activate unet-segmentation
```

For Mac M2 users, the environment is specifically configured with TensorFlow-macOS and Metal support.

### 2. Data Labeling with CVAT

#### a. Install Docker Desktop
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Sign up and open Docker Desktop

#### b. Install CVAT
```bash
# Clone CVAT repository
git clone https://github.com/openvinotoolkit/cvat.git

# Navigate to the CVAT directory
cd cvat

# Build with Docker
docker-compose build

# Start CVAT
docker-compose up -d
```

#### c. Access CVAT
- Open your browser and go to: http://localhost:8080/
- Create a new project
- Create a task and upload your images
- Open the job to see uploaded images
- Use the ellipse tool to label beads

### 3. Preparing Training Data

#### a. Combine Multiple Annotations (Team Work)
If multiple people are labeling data, use the `coco_combined.py` script to merge annotations:

```bash
python coco_combined.py
```

This will:
- Combine all COCO JSON files in the current directory
- Copy all images to a single directory (`image_combined/`)
- Create a combined annotation file (`combined.json`)

#### b. Generate Training Masks
Convert COCO annotations to different types of masks for U-Net training:

```bash
python generate_masks.py
```

This script generates:
- Binary masks (dataset/masks/)
- Circular masks (dataset/circular_masks/)
- Instance masks (dataset/instance_masks/)
- Boundary masks (dataset/boundary_masks/)

## Training the Model

### Instance Segmentation Training

Run the main training script:

```bash
python instance_segmentation_v5.py [options]
```

Options:
- `--max-images`: Maximum number of images to load (for testing)
- `--batch-size`: Batch size for training (default: 2)
- `--epochs`: Number of epochs to train (default: 30)
- `--filters`: Base number of filters in the U-Net (default: 32)
- `--image-size`: Image size for training (default: 256)
- `--skip-training`: Skip training and only do prediction/evaluation
- `--predict-only`: Only run prediction using existing model
- `--min-object-size`: Minimum object size for instance segmentation (default: 30)
- `--prepare-data-only`: Only prepare three-class dataset and exit
- `--visualize`: Create visualizations of the results (default: True)

Example:
```bash
python instance_segmentation_v5.py --epochs 50 --batch-size 4
```

## Prediction on New Images

### Predict Instance Segmentation

Use the trained model to predict on new images:

```bash
python prediction_v4.py [options]
```

Options:
- `--input-folder`: Folder containing new images (default: `new_images/`)
- `--output-folder`: Folder to save predictions (default: `predictions/`)
- `--model-path`: Path to trained model (default: `models/best_three_class_model.keras`)
- `--image-size`: Image size (default: 256)
- `--min-object-size`: Minimum object size (default: 30)
- `--visualize`: Create visualizations
- `--overlays`: Create prediction overlays on original images

Example:
```bash
python prediction_v4.py --input-folder ./new_images --output-folder ./predictions --visualize
```

## Output

### Training Outputs
- Trained model: `dataset/models/best_three_class_model.keras`
- Training history plot: `dataset/results/three_class_training_history.png`
- Prediction visualizations: `dataset/results/three_class_predictions_images_*.png`
- Instance segmentation results: `dataset/results/three_class_instance_predictions.npy`

### Prediction Outputs
- Instance masks: `predictions/instance_masks/`
- Prediction overlays: `predictions/overlays/`
- Visualization image: `predictions/prediction_visualization.png`
- All predictions as numpy array: `predictions/instance_predictions.npy`

## Model Architecture

The model uses an enhanced U-Net architecture with:
- Three-channel input (preprocessed with background subtraction, CLAHE, and distance transform)
- Multi-level encoder-decoder structure
- Skip connections between encoder and decoder
- Attention gates for better focus on boundaries
- Three-class output with softmax activation

### Loss Functions
- Combined loss: Weighted categorical cross-entropy + Dice loss
- Higher weights for boundary class to improve separation

### Metrics
- Accuracy
- Mean IoU
- Boundary accuracy

## Performance Notes

- For Mac M2 users: The code is configured to run in CPU mode for stability
- Memory management is optimized with batch processing and garbage collection
- The model saves checkpoints based on validation IoU

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Reduce image size
   - Use the `--max-images` option to limit dataset size during testing

2. **Model Loading Errors**
   - Ensure all custom objects are properly registered
   - Check if the model file exists in the specified path

3. **CVAT Installation Issues**
   - Make sure Docker is running before starting CVAT
   - Check Docker logs if containers fail to start

4. **TensorFlow Issues on Mac M2**
   - The environment is configured for Metal acceleration
   - If issues persist, ensure TensorFlow-macOS and TensorFlow-Metal are properly installed

## Citation

If you use this code in your research, please cite:

```
@software{unet_microbeads,
  author = {Your Name},
  title = {U-Net Microbeads Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/unet_microbeads}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the OpenVINO team for CVAT
- U-Net architecture inspired by Ronneberger et al.
- Instance segmentation approach based on watershed algorithm

## Contact

For questions or issues, please create an issue on the GitHub repository.