
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import gc

# Ensure we clean up memory whenever possible
gc.enable()

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")
CIRCULAR_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "circular_masks")
INSTANCE_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "instance_masks")
BOUNDARY_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "boundary_masks")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

# Ensure output folders exist
for folder in [MASK_FOLDER, CIRCULAR_MASK_FOLDER, INSTANCE_MASK_FOLDER,
               BOUNDARY_MASK_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def visualize_instance_mask(mask):
    """
    Visualize instance mask with different colors for each instance

    Args:
        mask: Instance mask with integer values

    Returns:
        colored_mask: RGB visualization of instance mask
    """
    # Create a colormap for instances (add 1 to max to include background)
    try:
        # Fix the matplotlib deprecation warning
        cmap = plt.colormaps['tab20']
        # If the above doesn't work, you can use the deprecated version:
        # cmap = plt.cm.get_cmap('tab20', np.max(mask) + 2)

        # Create RGB image (use the right number of colors)
        colored_mask = cmap(mask % 20)[:, :, :3]  # Drop alpha channel

        # Make background transparent
        colored_mask[mask == 0] = [0, 0, 0]

        return colored_mask
    except Exception as e:
        print(f"Error in visualize_instance_mask: {e}")
        # Return a simple colored version as fallback
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(1, np.max(mask) + 1):
            colored[mask == i] = [(i * 37) % 255, (i * 91) % 255, (i * 123) % 255]
        return colored

def generate_boundary_mask(instance_mask, thickness=2):
    """Generate boundary mask from instance mask"""
    boundaries = np.zeros_like(instance_mask, dtype=np.uint8)

    # For each instance
    for i in np.unique(instance_mask):
        if i == 0:  # Skip background
            continue

        # Get this instance
        binary = (instance_mask == i).astype(np.uint8)

        # Dilate it
        kernel = np.ones((thickness, thickness), np.uint8)
        dilated = cv2.dilate(binary, kernel)

        # Find boundaries (dilated - original)
        boundary = dilated & ~binary.astype(bool)

        # Add to boundary mask
        boundaries |= boundary

    return boundaries.astype(np.uint8) * 255

def generate_circular_masks():
    """
    Generate circular masks from COCO annotations
    """
    print("Generating circular masks from COCO annotations...")

    try:
        # Load COCO annotations
        with open(COCO_JSON_PATH, "r") as f:
            coco_data = json.load(f)

        # Create mappings
        image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco_data["images"]}

        # Group annotations by image ID
        image_annotations = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        # Process each image
        counter = 0
        for image_id, annotations in image_annotations.items():
            filename = image_id_to_filename.get(image_id)
            if not filename:
                continue

            width, height = image_id_to_size.get(image_id)

            # Create masks
            binary_mask = np.zeros((height, width), dtype=np.uint8)  # Standard binary mask
            circular_mask = np.zeros((height, width), dtype=np.uint8)  # Circle-fitted mask
            instance_mask = np.zeros((height, width), dtype=np.uint16)  # Instance mask

            # Process each annotation
            for idx, ann in enumerate(annotations, start=1):
                # For binary mask using original segmentation
                curr_mask = np.zeros((height, width), dtype=np.uint8)

                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):  # Polygon format
                        for segment in ann["segmentation"]:
                            pts = np.array(segment, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(curr_mask, [pts], color=255)
                    elif isinstance(ann["segmentation"], dict):  # RLE format
                        try:
                            from pycocotools import mask as mask_utils
                            rle = ann["segmentation"]
                            if isinstance(rle["counts"], list):  # Convert list to valid RLE format
                                rle = mask_utils.frPyObjects([rle], height, width)
                                rle = mask_utils.merge(rle)
                            curr_mask = mask_utils.decode(rle).astype(np.uint8) * 255
                        except ImportError:
                            print("Warning: pycocotools not found. Skipping RLE masks.")
                            continue

                # Add to binary mask
                binary_mask = np.maximum(binary_mask, curr_mask)

                # Add to instance mask
                instance_mask[curr_mask > 0] = idx

                # Generate circular mask from bbox or segmentation
                if "bbox" in ann:
                    # Use bounding box for circle center and radius
                    x, y, w, h = [int(v) for v in ann["bbox"]]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    radius = max(w, h) // 2  # Use the larger dimension
                else:
                    # Use segmentation to find center and approximate radius
                    contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Find the largest contour
                        contour = max(contours, key=cv2.contourArea)
                        # Find centroid
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            # Estimate radius from area
                            area = cv2.contourArea(contour)
                            radius = int(np.sqrt(area / np.pi))
                        else:
                            # Skip if can't compute centroid
                            continue
                    else:
                        # Skip if no contour found
                        continue

                # Draw circle on circular mask
                cv2.circle(circular_mask, (center_x, center_y), radius, 255, -1)

            # Generate boundary mask from instance mask
            boundary_mask = generate_boundary_mask(instance_mask)

            # Save all masks
            cv2.imwrite(os.path.join(MASK_FOLDER, filename), binary_mask)
            cv2.imwrite(os.path.join(CIRCULAR_MASK_FOLDER, filename), circular_mask)

            # Save instance mask as uint16 PNG
            instance_filename = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(INSTANCE_MASK_FOLDER, instance_filename), instance_mask)

            # Debug: Check instance mask values
            test_mask = cv2.imread(os.path.join(INSTANCE_MASK_FOLDER, instance_filename), cv2.IMREAD_UNCHANGED)
            print(f"Instance mask values: min={np.min(test_mask)}, max={np.max(test_mask)}, unique={np.unique(test_mask)}")

            # Create visualization of instance mask
            colored_mask = visualize_instance_mask(instance_mask)
            plt.figure(figsize=(8, 8))
            plt.imshow(colored_mask)
            plt.title(f"Instance Mask - {filename}")
            plt.axis('off')
            plt.savefig(os.path.join(RESULTS_FOLDER, f"instance_visualization_{os.path.splitext(filename)[0]}.png"))
            plt.close()

            # Save boundary mask
            cv2.imwrite(os.path.join(BOUNDARY_MASK_FOLDER, filename), boundary_mask)

            # Increment counter and clean up memory every 10 images
            counter += 1
            if counter % 10 == 0:
                gc.collect()

        print(f"Generated masks for {len(image_annotations)} images")
        print(f"- Binary masks: {MASK_FOLDER}")
        print(f"- Circular masks: {CIRCULAR_MASK_FOLDER}")
        print(f"- Instance masks: {INSTANCE_MASK_FOLDER}")
        print(f"- Boundary masks: {BOUNDARY_MASK_FOLDER}")
        print(f"- Instance visualizations: {RESULTS_FOLDER}")

    except Exception as e:
        print(f"Error in generate_circular_masks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("MASK GENERATION SCRIPT")
    print("=" * 50)

    # Generate masks
    generate_circular_masks()

    # Final cleanup
    gc.collect()

    print("Mask generation complete!")
