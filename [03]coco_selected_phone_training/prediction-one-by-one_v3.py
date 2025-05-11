
#!/usr/bin/env python3
"""
Improved prediction script that processes images one at a time,
with consistent preprocessing matching the training code.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import gc
import logging
import time
import traceback
from skimage import morphology
from scipy import ndimage as ndi
from skimage import feature
from skimage.segmentation import watershed
from skimage.measure import regionprops, label as skimage_label

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_improved.log"),
        logging.StreamHandler()
    ]
)

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.info("Running in CPU-only mode for stability")

# Import from the original training script
from instance_segmentation_v2 import (
    NormalizationLayer,
    dice_loss,
    f1_score,
    iou_score,
    preprocess_image  # Import the EXACT preprocessing function used during training
)

# Paths
INPUT_DIR = "new_images"
OUTPUT_DIR = "predictions_improved"
MODEL_PATH = "models/best_binary_model.keras"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "debug"), exist_ok=True)

def extract_instances_from_binary(binary_mask, min_size=30, min_distance=8):
    """
    Extract individual instances from a binary mask using improved watershed
    with better handling of clustered beads

    Args:
        binary_mask: Binary mask (foreground/background)
        min_size: Minimum size of objects to keep
        min_distance: Minimum distance between markers (smaller value helps separate clusters)

    Returns:
        Instance mask with unique IDs for each object
    """
    try:
        # Ensure binary mask is boolean
        binary = binary_mask.astype(bool)

        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=min_size)

        # Apply morphological operations to help separate touching beads
        # This can help break weak connections between beads
        kernel = np.ones((3,3), np.uint8)
        binary_opened = morphology.opening(binary, kernel)

        # Distance transform - important for watershed seeding
        distance = ndi.distance_transform_edt(binary_opened)

        # Enhance the distance transform to better separate clustered beads
        # Apply Gaussian blur to smooth it slightly
        distance_smoothed = ndi.gaussian_filter(distance, sigma=1.0)

        # Find local maxima with a smaller min_distance parameter
        # This helps detect more seeds in clustered areas
        coords = feature.peak_local_max(
            distance_smoothed,
            min_distance=min_distance,  # Reduced from the default to better separate clusters
            threshold_abs=0.1,         # Lower threshold to detect more peaks
            labels=binary_opened       # Use the opened binary mask
        )

        # Create markers for watershed
        markers = np.zeros_like(binary, dtype=np.int32)
        if len(coords) > 0:
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

            # Apply watershed with the negative distance transform
            labeled_mask = watershed(-distance_smoothed, markers, mask=binary)
        else:
            # Fallback to connected components if no peaks found
            labeled_mask = morphology.label(binary)

        # Post-process: merge very small regions into adjacent larger ones
        # This helps clean up over-segmentation while keeping properly separated beads
        props = regionprops(labeled_mask)

        # Identify very small regions to potentially merge
        small_regions = []
        for prop in props:
            if prop.area < min_size * 0.5:  # Regions less than half of min_size
                small_regions.append(prop.label)

        # Merge small regions if needed
        if small_regions:
            for small_label in small_regions:
                # Find boundaries of this region
                small_mask = labeled_mask == small_label
                dilated = morphology.binary_dilation(small_mask)
                neighbors = labeled_mask[dilated & ~small_mask]

                # Find most common neighbor (excluding background and itself)
                neighbors = neighbors[neighbors > 0]
                neighbors = neighbors[neighbors != small_label]

                if len(neighbors) > 0:
                    # Merge with most common neighbor
                    most_common = np.bincount(neighbors).argmax()
                    labeled_mask[small_mask] = most_common

        return labeled_mask
    except Exception as e:
        logging.error(f"Error in extract_instances_from_binary: {e}")
        traceback.print_exc()
        # Fallback to basic connected components
        return morphology.label(binary)

def create_instance_overlay(image, instance_mask):
    """
    Create a colored overlay of instance segmentation on the original image

    Args:
        image: Original grayscale image
        instance_mask: Instance segmentation mask with unique IDs

    Returns:
        Colored overlay image
    """
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        rgb_img = np.stack([image, image, image], axis=-1)
    else:
        rgb_img = image

    # Make sure we're working with uint8
    if rgb_img.dtype != np.uint8:
        rgb_img = (rgb_img * 255).astype(np.uint8)

    # Create overlay
    overlay = rgb_img.copy()

    # Get unique instance IDs (excluding background = 0)
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances > 0]

    # No instances found
    if len(unique_instances) == 0:
        return rgb_img

    # Assign colors to each instance
    for i, instance_id in enumerate(unique_instances):
        # Generate a distinct color for each instance
        color = np.array([
            (instance_id * 50) % 255,
            (instance_id * 100) % 255,
            (instance_id * 150) % 255
        ], dtype=np.uint8)

        # Create mask for this instance
        instance_pixels = (instance_mask == instance_id)

        # Apply color with alpha blending (50% original, 50% color)
        overlay[instance_pixels] = (overlay[instance_pixels] * 0.5 + color * 0.5).astype(np.uint8)

        # Draw contour around the instance
        contours, _ = cv2.findContours(
            instance_pixels.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay

def process_single_image(image_path):
    """Process a single image with enhanced instance segmentation"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Normalize to 0-1
        image = image.astype(np.float32) / 255.0
        
        # Preprocess image
        preprocessed = preprocess_image(image)
        
        # Add batch dimension
        input_tensor = np.expand_dims(preprocessed, 0)
        
        # Get predictions
        binary_pred, boundary_pred = model.predict(input_tensor)
        
        # Remove batch dimension
        binary_pred = binary_pred[0, ..., 0]
        boundary_pred = boundary_pred[0, ..., 0]
        
        # Threshold predictions
        binary_mask = (binary_pred > 0.5).astype(np.uint8)
        boundary_mask = (boundary_pred > 0.5).astype(np.uint8)
        
        # Use boundary information to improve instance separation
        # Reduce the binary mask values where boundaries are detected
        binary_mask_float = binary_mask.astype(np.float32)
        binary_mask_float[boundary_mask > 0] *= 0.3
        
        # Extract instances with boundary awareness
        instance_mask = extract_instances_from_binary(binary_mask_float)
        
        return image, binary_mask, instance_mask, boundary_mask
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        traceback.print_exc()
        return None, None, None, None

def main():
    """Process all images one by one"""
    # Get all images
    image_paths = glob(os.path.join(INPUT_DIR, "*.jpg")) + glob(os.path.join(INPUT_DIR, "*.png"))
    if not image_paths:
        logging.error(f"No images found in {INPUT_DIR}")
        return

    logging.info(f"Found {len(image_paths)} images to process")

    # Process each image separately
    successful = 0
    for i, image_path in enumerate(image_paths):
        logging.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        start_time = time.time()

        # Process image
        success = process_single_image(image_path)

        elapsed = time.time() - start_time
        if success:
            successful += 1
            logging.info(f"Successfully processed image in {elapsed:.2f} seconds")
        else:
            logging.warning(f"Failed to process image after {elapsed:.2f} seconds")

        # Always force cleanup between images
        plt.close('all')
        gc.collect()

    logging.info(f"All images processed. Success rate: {successful}/{len(image_paths)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    logging.info(f"Total execution time: {elapsed:.2f} seconds")

# Add boundary awareness to better separate touching beads
edge_map = feature.canny(binary_opened, sigma=2.0)

# Combine distance transform with edge information
distance_edge = distance_smoothed.copy()
distance_edge[edge_map] *= 0.5  # Reduce distance at edges

# Use adaptive thresholding for peak detection
threshold = np.mean(distance_edge[binary_opened]) * 0.3

# Find local maxima with adaptive parameters
coords = feature.peak_local_max(
    distance_edge,
    min_distance=min_distance,
    threshold_abs=threshold,
    labels=binary_opened,
    exclude_border=False
)
