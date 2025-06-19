
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
import pandas as pd
from PIL import Image
import argparse

def process_image_for_training_data(image_path, sigma_value=5, threshold_value=0.02,
                                   min_radius=1.0, max_radius=10.0, show_visualization=False):
    """
    Process a single image to generate segmentation mask and extract centroids using blob_dog

    Args:
        image_path (str): Path to the input image
        sigma_value (float): Max sigma for blob_dog detection
        threshold_value (float): Threshold for blob_dog detection
        min_radius (float): Minimum radius for filtering blobs
        max_radius (float): Maximum radius for filtering blobs
        show_visualization (bool): Whether to show visualization

    Returns:
        tuple: (binary_mask, centroids_data, visualization_image)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]

    # Process with inversion (based on your optimized parameters)
    inverted_img = cv2.bitwise_not(img)
    blurred_img_invert = cv2.GaussianBlur(cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY), (5, 5), 3)

    # Blob detection (DoG) with your optimized parameters
    blobs_dog = blob_dog(blurred_img_invert, max_sigma=sigma_value, threshold=threshold_value)

    # Filter blobs by radius
    filtered_blobs = []
    for blob in blobs_dog:
        y, x, r = blob
        if min_radius <= r <= max_radius:
            # Ensure coordinates are within image bounds
            if 0 <= x < width and 0 <= y < height:
                filtered_blobs.append([y, x, r])

    filtered_blobs = np.array(filtered_blobs) if filtered_blobs else np.empty((0, 3))

    # Create binary segmentation mask
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    centroids_data = []

    # Get base filename for image_id
    base_filename = os.path.basename(image_path)
    image_id = os.path.splitext(base_filename)[0]

    for i, blob in enumerate(filtered_blobs):
        y, x, r = blob

        # Draw filled circle on mask
        cv2.circle(binary_mask, (int(x), int(y)), int(r), 255, -1)

        # Store centroid data
        centroids_data.append({
            'image_id': f"{image_id}_{i}",
            'image_filename': base_filename,
            'x': float(x),
            'y': float(y),
            'approx_radius': float(r),
            'blob_index': i
        })

    # Create visualization if requested
    visualization_img = None
    if show_visualization:
        visualization_img = img.copy()
        for blob in filtered_blobs:
            y, x, r = blob
            cv2.circle(visualization_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(visualization_img, (int(x), int(y)), 2, (0, 0, 255), -1)  # Centroid dot

    return binary_mask, centroids_data, visualization_img

def process_dataset(input_folder, output_folder, sigma_value=5, threshold_value=0.02,
                   min_radius=1.0, max_radius=10.0, show_visualizations=False):
    """
    Process entire dataset to generate masks and centroids CSV

    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path to output folder
        sigma_value (float): Max sigma for blob_dog detection
        threshold_value (float): Threshold for blob_dog detection
        min_radius (float): Minimum radius for filtering blobs
        max_radius (float): Maximum radius for filtering blobs
        show_visualizations (bool): Whether to save visualization images
    """

    # Create output directories
    masks_folder = os.path.join(output_folder, 'masks')
    visualizations_folder = os.path.join(output_folder, 'visualizations')

    os.makedirs(masks_folder, exist_ok=True)
    if show_visualizations:
        os.makedirs(visualizations_folder, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    # Get all image files
    image_files = []
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    image_files.sort()
    print(f"Found {len(image_files)} images to process")

    # Process each image
    all_centroids = []
    processed_count = 0
    failed_count = 0

    for image_file in image_files:
        try:
            image_path = os.path.join(input_folder, image_file)
            print(f"Processing: {image_file}")

            # Process image
            binary_mask, centroids_data, visualization_img = process_image_for_training_data(
                image_path, sigma_value, threshold_value, min_radius, max_radius, show_visualizations
            )

            # Save binary mask
            base_name = os.path.splitext(image_file)[0]
            mask_filename = f"{base_name}.png"
            mask_path = os.path.join(masks_folder, mask_filename)
            cv2.imwrite(mask_path, binary_mask)

            # Save visualization if requested
            if show_visualizations and visualization_img is not None:
                vis_filename = f"{base_name}_visualization.png"
                vis_path = os.path.join(visualizations_folder, vis_filename)
                cv2.imwrite(vis_path, visualization_img)

            # Collect centroid data
            all_centroids.extend(centroids_data)

            print(f"  - Found {len(centroids_data)} particles")
            processed_count += 1

        except Exception as e:
            print(f"  - ERROR processing {image_file}: {str(e)}")
            failed_count += 1
            continue

    # Save centroids CSV
    if all_centroids:
        centroids_df = pd.DataFrame(all_centroids)
        csv_path = os.path.join(output_folder, 'all_centroids_copy.csv')
        centroids_df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(all_centroids)} centroids to: {csv_path}")
    else:
        print("\nWarning: No centroids detected in any images!")

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"- Successfully processed: {processed_count} images")
    print(f"- Failed: {failed_count} images")
    print(f"- Total particles detected: {len(all_centroids)}")
    print(f"- Average particles per image: {len(all_centroids)/max(processed_count, 1):.2f}")

    # Print parameter summary
    print(f"\nParameters used:")
    print(f"- Sigma value: {sigma_value}")
    print(f"- Threshold: {threshold_value}")
    print(f"- Radius range: {min_radius} - {max_radius}")

    return all_centroids

def analyze_detection_quality(csv_path, visualizations_folder=None):
    """
    Analyze the quality of detections and provide statistics
    """
    if not os.path.exists(csv_path):
        print("CSV file not found for analysis")
        return

    df = pd.read_csv(csv_path)

    print("\n" + "="*50)
    print("DETECTION QUALITY ANALYSIS")
    print("="*50)

    # Basic statistics
    print(f"Total detections: {len(df)}")
    print(f"Unique images: {df['image_filename'].nunique()}")
    print(f"Average detections per image: {len(df) / df['image_filename'].nunique():.2f}")

    # Radius distribution
    print(f"\nRadius distribution:")
    print(f"- Mean radius: {df['approx_radius'].mean():.2f}")
    print(f"- Median radius: {df['approx_radius'].median():.2f}")
    print(f"- Min radius: {df['approx_radius'].min():.2f}")
    print(f"- Max radius: {df['approx_radius'].max():.2f}")

    # Per-image statistics
    per_image_stats = df.groupby('image_filename')['approx_radius'].agg(['count', 'mean', 'std']).round(2)
    per_image_stats.columns = ['particle_count', 'avg_radius', 'radius_std']

    print(f"\nPer-image statistics:")
    print(f"- Images with 0 particles: {(per_image_stats['particle_count'] == 0).sum()}")
    print(f"- Images with 1-5 particles: {((per_image_stats['particle_count'] >= 1) & (per_image_stats['particle_count'] <= 5)).sum()}")
    print(f"- Images with 6-20 particles: {((per_image_stats['particle_count'] >= 6) & (per_image_stats['particle_count'] <= 20)).sum()}")
    print(f"- Images with >20 particles: {(per_image_stats['particle_count'] > 20).sum()}")

def main():
    parser = argparse.ArgumentParser(description='Generate training masks and centroids using blob_dog detection')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory containing input images')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Path to output folder for masks and CSV (default: blob_dog_masks_centroid in input_dir)')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Max sigma for blob_dog detection (default: 5.0)')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Threshold for blob_dog detection (default: 0.02)')
    parser.add_argument('--min_radius', type=float, default=1.0,
                        help='Minimum radius for filtering blobs (default: 1.0)')
    parser.add_argument('--max_radius', type=float, default=10.0,
                        help='Maximum radius for filtering blobs (default: 10.0)')
    parser.add_argument('--visualizations', action='store_true',
                        help='Save visualization images showing detected blobs')
    parser.add_argument('--analyze', action='store_true',
                        help='Run quality analysis after processing')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return

    # Set default output folder if not provided
    if args.output_folder is None:
        args.output_folder = os.path.join(args.input_dir, 'blob_dog_masks_centroid')

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Processing images from: {args.input_dir}")
    print(f"Output will be saved to: {args.output_folder}")
    print(f"Parameters: sigma={args.sigma}, threshold={args.threshold}, radius_range=[{args.min_radius}, {args.max_radius}]")

    # Process dataset
    centroids = process_dataset(
        input_folder=args.input_dir,
        output_folder=args.output_folder,
        sigma_value=args.sigma,
        threshold_value=args.threshold,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        show_visualizations=args.visualizations
    )

    # Run analysis if requested
    if args.analyze:
        csv_path = os.path.join(args.output_folder, 'all_centroids_copy.csv')
        analyze_detection_quality(csv_path)

    print(f"\nGenerated files:")
    print(f"- Masks: {os.path.join(args.output_folder, 'masks/')}")
    print(f"- Centroids CSV: {os.path.join(args.output_folder, 'all_centroids_copy.csv')}")
    if args.visualizations:
        print(f"- Visualizations: {os.path.join(args.output_folder, 'visualizations/')}")

if __name__ == "__main__":
    main()
