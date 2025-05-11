
import os
import json
import numpy as np
import cv2
import pandas as pd
from glob import glob
import gc

# Ensure we clean up memory whenever possible
gc.enable()

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")

# Ensure output folders exist
for folder in [RESULTS_FOLDER, CENTROIDS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def extract_centroids_from_coco():
    """
    Extract and export centroids from COCO annotations
    """
    print("Extracting centroids from COCO annotations...")

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

        # Create a list to store all centroids with metadata
        all_centroids = []

        # Process each image
        counter = 0
        for image_id, annotations in image_annotations.items():
            filename = image_id_to_filename.get(image_id)
            if not filename:
                continue

            width, height = image_id_to_size.get(image_id)

            # List to store centroids for this image
            image_centroids = []

            # Process each annotation
            for idx, ann in enumerate(annotations, start=1):
                # Create temp mask for this annotation
                curr_mask = np.zeros((height, width), dtype=np.uint8)

                # Get bead ID if available
                bead_id = ann.get("id", idx)

                # Get category ID if available
                category_id = ann.get("category_id", 1)

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

                # Method 1: Use contour to find centroid
                contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Find the largest contour
                    contour = max(contours, key=cv2.contourArea)
                    # Find centroid using moments
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        # Get area and approximate radius
                        area = cv2.contourArea(contour)
                        approx_radius = int(np.sqrt(area / np.pi))

                        # Store centroid with metadata
                        centroid_data = {
                            "image_id": image_id,
                            "image_filename": filename,
                            "bead_id": bead_id,
                            "category_id": category_id,
                            "x": center_x,
                            "y": center_y,
                            "area": area,
                            "approx_radius": approx_radius
                        }

                        image_centroids.append(centroid_data)
                        all_centroids.append(centroid_data)
                    else:
                        print(f"Warning: Could not compute centroid for bead {bead_id} in image {filename}")
                else:
                    # Method 2: Use bbox if available as fallback
                    if "bbox" in ann:
                        x, y, w, h = [int(v) for v in ann["bbox"]]
                        center_x = x + w // 2
                        center_y = y + h // 2
                        area = w * h
                        approx_radius = max(w, h) // 2

                        # Store centroid with metadata
                        centroid_data = {
                            "image_id": image_id,
                            "image_filename": filename,
                            "bead_id": bead_id,
                            "category_id": category_id,
                            "x": center_x,
                            "y": center_y,
                            "area": area,
                            "approx_radius": approx_radius,
                            "note": "calculated_from_bbox"
                        }

                        image_centroids.append(centroid_data)
                        all_centroids.append(centroid_data)
                    else:
                        print(f"Warning: No valid contour or bbox found for bead {bead_id} in image {filename}")

            # Save centroids for this image as CSV
            if image_centroids:
                image_centroids_df = pd.DataFrame(image_centroids)
                csv_filename = os.path.splitext(filename)[0] + "_centroids.csv"
                image_centroids_df.to_csv(os.path.join(CENTROIDS_FOLDER, csv_filename), index=False)

                # Create a visualization of centroids on a blank image
                visual_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                for centroid in image_centroids:
                    x, y = centroid["x"], centroid["y"]
                    radius = centroid["approx_radius"]
                    # Draw circle at centroid position
                    cv2.circle(visual_img, (x, y), radius, (0, 0, 255), 2)
                    # Draw point at exact centroid
                    cv2.circle(visual_img, (x, y), 3, (255, 0, 0), -1)
                    # Add ID text
                    cv2.putText(visual_img, str(centroid["bead_id"]), (x+5, y+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Save visualization
                viz_filename = os.path.splitext(filename)[0] + "_centroids_viz.png"
                cv2.imwrite(os.path.join(RESULTS_FOLDER, viz_filename), visual_img)

            # Increment counter and clean up memory every 10 images
            counter += 1
            if counter % 10 == 0:
                gc.collect()

        # Save all centroids as a single CSV
        if all_centroids:
            all_centroids_df = pd.DataFrame(all_centroids)
            all_centroids_df.to_csv(os.path.join(RESULTS_FOLDER, "all_centroids.csv"), index=False)

            # Generate summary statistics
            print(f"Total centroids extracted: {len(all_centroids)}")
            print(f"Centroids per image:")
            image_counts = all_centroids_df.groupby('image_filename').size()
            print(image_counts.describe())

            # Save summary statistics
            with open(os.path.join(RESULTS_FOLDER, "centroids_summary.txt"), "w") as f:
                f.write(f"Total centroids extracted: {len(all_centroids)}\n")
                f.write(f"Centroids per image:\n")
                f.write(str(image_counts.describe()))

        print(f"Extracted centroids for {len(image_annotations)} images")
        print(f"- Individual image centroids: {CENTROIDS_FOLDER}")
        print(f"- All centroids combined: {os.path.join(RESULTS_FOLDER, 'all_centroids.csv')}")
        print(f"- Centroid visualizations: {RESULTS_FOLDER}")

    except Exception as e:
        print(f"Error in extract_centroids_from_coco: {e}")
        import traceback
        traceback.print_exc()

def export_centroids_in_yolo_format():
    """
    Export centroids in YOLO format (normalized coordinates)
    """
    print("Exporting centroids in YOLO format...")

    try:
        # Load COCO annotations
        with open(COCO_JSON_PATH, "r") as f:
            coco_data = json.load(f)

        # Create mappings
        image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco_data["images"]}

        # Create YOLO labels folder
        YOLO_LABELS_FOLDER = os.path.join(BASE_DIR, "dataset", "yolo_labels")
        os.makedirs(YOLO_LABELS_FOLDER, exist_ok=True)

        # Group annotations by image ID
        image_annotations = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        # Process each image
        for image_id, annotations in image_annotations.items():
            filename = image_id_to_filename.get(image_id)
            if not filename:
                continue

            width, height = image_id_to_size.get(image_id)

            # Create a txt file for YOLO format (same name as image but .txt extension)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(YOLO_LABELS_FOLDER, txt_filename)

            with open(txt_path, "w") as f:
                # Process each annotation
                for ann in annotations:
                    # Get category ID (default to 0 if not present)
                    category_id = ann.get("category_id", 0)

                    # Calculate centroid
                    if "segmentation" in ann:
                        # Create temp mask for this annotation
                        curr_mask = np.zeros((height, width), dtype=np.uint8)

                        if isinstance(ann["segmentation"], list):  # Polygon format
                            for segment in ann["segmentation"]:
                                pts = np.array(segment, dtype=np.int32).reshape(-1, 2)
                                cv2.fillPoly(curr_mask, [pts], color=255)
                        elif isinstance(ann["segmentation"], dict):  # RLE format
                            try:
                                from pycocotools import mask as mask_utils
                                rle = ann["segmentation"]
                                if isinstance(rle["counts"], list):
                                    rle = mask_utils.frPyObjects([rle], height, width)
                                    rle = mask_utils.merge(rle)
                                curr_mask = mask_utils.decode(rle).astype(np.uint8) * 255
                            except ImportError:
                                # Use bbox as fallback
                                if "bbox" in ann:
                                    x, y, w, h = [float(v) for v in ann["bbox"]]
                                    # Normalize coordinates
                                    center_x = (x + w/2) / width
                                    center_y = (y + h/2) / height
                                    # Write YOLO format: class x_center y_center width height
                                    f.write(f"{category_id} {center_x:.6f} {center_y:.6f} {w/width:.6f} {h/height:.6f}\n")
                                continue

                        # Find contours and centroid
                        contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Find the largest contour
                            contour = max(contours, key=cv2.contourArea)
                            # Find centroid
                            M = cv2.moments(contour)
                            if M["m00"] > 0:
                                center_x = M["m10"] / M["m00"]
                                center_y = M["m01"] / M["m00"]

                                # Calculate approximate radius from area
                                area = cv2.contourArea(contour)
                                radius = np.sqrt(area / np.pi)

                                # Normalize coordinates
                                norm_center_x = center_x / width
                                norm_center_y = center_y / height
                                norm_width = (radius * 2) / width
                                norm_height = (radius * 2) / height

                                # Write YOLO format: class x_center y_center width height
                                f.write(f"{category_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                            else:
                                # Use bbox as fallback
                                if "bbox" in ann:
                                    x, y, w, h = [float(v) for v in ann["bbox"]]
                                    # Normalize coordinates
                                    center_x = (x + w/2) / width
                                    center_y = (y + h/2) / height
                                    # Write YOLO format: class x_center y_center width height
                                    f.write(f"{category_id} {center_x:.6f} {center_y:.6f} {w/width:.6f} {h/height:.6f}\n")
                        else:
                            # Use bbox as fallback
                            if "bbox" in ann:
                                x, y, w, h = [float(v) for v in ann["bbox"]]
                                # Normalize coordinates
                                center_x = (x + w/2) / width
                                center_y = (y + h/2) / height
                                # Write YOLO format: class x_center y_center width height
                                f.write(f"{category_id} {center_x:.6f} {center_y:.6f} {w/width:.6f} {h/height:.6f}\n")
                    elif "bbox" in ann:
                        # If only bbox is available
                        x, y, w, h = [float(v) for v in ann["bbox"]]
                        # Normalize coordinates
                        center_x = (x + w/2) / width
                        center_y = (y + h/2) / height
                        # Write YOLO format: class x_center y_center width height
                        f.write(f"{category_id} {center_x:.6f} {center_y:.6f} {w/width:.6f} {h/height:.6f}\n")

        print(f"Exported YOLO-format centroid labels to {YOLO_LABELS_FOLDER}")

    except Exception as e:
        print(f"Error in export_centroids_in_yolo_format: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("CENTROID EXTRACTION SCRIPT")
    print("=" * 50)

    # Extract centroids and save as CSV
    extract_centroids_from_coco()

    # Export in YOLO format
    export_centroids_in_yolo_format()

    # Final cleanup
    gc.collect()

    print("Centroid extraction complete!")
