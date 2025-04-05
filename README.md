# U-Net Microbeads Detection
unet_microbeads/
├── .git/                            # Git repository data
├── README.md                        # This file
├── coco_selected_microscopy_training/   # Training data for microscopy images
├── coco_selected_phone_training/    # Training data for smartphone images
├── dataset/                         # Main dataset directory
│   ├── results/                     # Results from model predictions
│   ├── __pycache__/                 # Python cache files
│   ├── annotations.json             # COCO format annotations
│   ├── generate_masks.py            # Script to generate masks from annotations
│   ├── instance_segmentation_v2.py  # Instance segmentation implementation
│   ├── models/                      # Saved model weights
│   └── prediction_improved.log      # Prediction logs
├── environment_unet.yml             # Conda environment file
└── other directories...             # Additional project directories
