
import os
import shutil
import pandas as pd
import re
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob

def get_dilution_factor(folder_name):
    """
    Extract dilution factor from folder name
    This assumes the dilution factor is stored in a consistent way
    """
    if folder_name.endswith('X'):
        return folder_name
    parts = folder_name.split('_')
    if len(parts) > 1:
        return parts[-1]  # Return the last part after underscore
    return folder_name    # Return the whole name if no underscore

def organize_images(input_dir=None, output_dir=None, density_csv=None):
    """
    Organize images into a single folder and create a CSV with density mappings.

    Args:
        input_dir: Input directory containing microscopy images
        output_dir: Output directory for organized images
        density_csv: Path to CSV containing density values for different dilutions
    """
    # Set up paths
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # If input_dir not provided, ask user
    if input_dir is None:
        print("Enter the folder name containing the microscopy images:")
        input_dir = input().strip()

    # Set output directory for original images
    if output_dir is None:
        images_folder = os.path.join(dataset_dir, 'original_images')
    else:
        images_folder = output_dir
    os.makedirs(images_folder, exist_ok=True)

    # Load predicted densities with the updated column name
    if density_csv is None:
        predicted_densities_path = os.path.join(dataset_dir, 'predicted_densities.csv')
    else:
        predicted_densities_path = density_csv

    try:
        predicted_densities_df = pd.read_csv(predicted_densities_path)
        print(f"CSV columns: {predicted_densities_df.columns.tolist()}")

        # Create a dictionary mapping dilution factors to predicted densities
        # Convert all keys to uppercase to match folder names
        density_dict = {k.upper(): v for k, v in zip(
            predicted_densities_df['Dilution factor'],
            predicted_densities_df['Predicted_Density']
        )}
        print(f"Density dictionary: {density_dict}")
    except Exception as e:
        print(f"Warning: Could not load density CSV: {e}")
        print("Continuing without density values. You can add them later.")
        density_dict = {}

    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The folder '{input_dir}' does not exist.")
        return

    # Initialize lists for the new CSV
    folder_filename = []
    filename = []
    predicted_density = []

    # Find all image files in the input directory and its subdirectories
    image_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
    image_files = []

    # First, process files directly in the input directory
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))

    # Then process subdirectories (if input_dir contains dilution factor folders)
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        # Skip if not a directory or if it's a special directory
        if not os.path.isdir(folder_path) or folder_name in [
            '.ipynb_checkpoints', 'original_images'
        ]:
            continue

        # Try to get dilution factor from folder name
        dilution_factor = folder_name if folder_name.endswith('X') else None

        # Process each image in the folder
        for ext in image_extensions:
            for file_path in glob.glob(os.path.join(folder_path, f'*{ext}')):
                image_files.append(file_path)

    print(f"Found {len(image_files)} image files")

    # Process each image file
    for image_path in image_files:
        try:
            # Get file name and folder name
            file = os.path.basename(image_path)
            folder_name = os.path.basename(os.path.dirname(image_path))

            # Try to determine dilution factor
            dilution_factor = None

            # First, check if folder name is a dilution factor
            if folder_name.endswith('X'):
                dilution_factor = folder_name

            # If not, try to extract from filename (e.g., "10X_image.tif")
            if dilution_factor is None:
                match = re.match(r'(\d+X)_', file)
                if match:
                    dilution_factor = match.group(1)

            # If still not found, try other extraction methods based on your naming convention
            if dilution_factor is None:
                # Example: extract "80x" from "1_80x stock m270_4xobj_1.tif"
                match = re.search(r'_(\d+x)_', file) or re.search(r'_(\d+x)', file)
                if match:
                    dilution_factor = match.group(1).upper()  # Convert to uppercase

            # Default value if no dilution factor found
            if dilution_factor is None:
                dilution_factor = "UNKNOWN"
                print(f"Warning: Could not determine dilution factor for: {file}")

            # Create a new filename with folder prefix (if applicable)
            if folder_name.endswith('X'):
                # If image is in a dilution folder, prefix with folder name
                new_filename = f"{folder_name}_{file}"
            else:
                # Otherwise, just use the original filename
                new_filename = file

            # Copy the file to the images directory
            dst_path = os.path.join(images_folder, new_filename)
            shutil.copy2(image_path, dst_path)
            print(f"Copied: {image_path} -> {dst_path}")

            # Get density if available
            density_value = None
            if dilution_factor in density_dict:
                density_value = density_dict[dilution_factor]

            # Add to our lists for the CSV
            folder_filename.append(new_filename)
            filename.append(file)
            predicted_density.append(density_value)

            print(f"Processed {file} with dilution factor: {dilution_factor}, density: {density_value}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Create and save the CSV file
    df = pd.DataFrame({
        'folderName_filename': folder_filename,
        'filename': filename,
        'predicted_density': predicted_density
    })

    # Sort by dilution factor
    def extract_number(x):
        # Extract the dilution factor part (folder name or filename)
        match = re.search(r'(\d+)X', x)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0
        return 0  # Default value if no match

    # Sort by the numeric part of the dilution factor
    if not df.empty:
        df['sort_key'] = df['folderName_filename'].apply(extract_number)
        df = df.sort_values('sort_key')
        df = df.drop('sort_key', axis=1)
    else:
        print("Warning: No images were processed. The DataFrame is empty.")

    # Save the CSV
    image_density_csv_path = os.path.join(images_folder, 'image_density_mapping.csv')
    df.to_csv(image_density_csv_path, index=False)

    print(f"Successfully processed {len(df)} images")
    print(f"Images copied to: {images_folder}")
    print(f"CSV file created at: {image_density_csv_path}")

    # Create the bead density CSV in the expected format
    add_filenames_to_density_csv(df, dataset_dir)

    return df

def add_filenames_to_density_csv(image_df, dataset_dir):
    """
    Create a CSV file with filenames and density values.

    Parameters:
    -----------
    image_df : DataFrame
        DataFrame containing image filenames and density values
    dataset_dir : str
        Path to the dataset directory
    """
    # Directory paths
    output_csv_path = os.path.join(dataset_dir, 'beadDensity_with_filenames.csv')

    # Handle empty or null density values
    image_df['predicted_density'] = image_df['predicted_density'].fillna(-1)  # Fill NA with -1 to indicate missing

    # Prepare the result DataFrame
    filenames_without_extension = [os.path.splitext(file)[0] for file in image_df['folderName_filename']]

    result_df = pd.DataFrame({
        'filename': filenames_without_extension,
        'density': image_df['predicted_density']
    })

    # Save the new DataFrame to a CSV file with UTF-8 encoding
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Bead density CSV file created at {output_csv_path}")

    return result_df

def crop_center_grid(dataset_dir=None, grid_size=7, crop_size=512, overlap=0, visualize=True):
    """
    Creates a grid of crops from the center of each image, excluding the 4 corner cells.
    This version handles high-resolution microscopy images (3840x2160) with more crops.

    Args:
        dataset_dir: Dataset directory containing original_images folder
        grid_size: Size of the grid (e.g., 7 for a 7x7 grid)
        crop_size: Size of each crop in pixels (default 512)
        overlap: Overlap between adjacent crops in pixels (default 0)
        visualize: Whether to create visualization images
    """
    # Set up paths
    if dataset_dir is None:
        root_dir = os.getcwd()
        dataset_dir = os.path.join(root_dir, 'dataset')

    input_folder = os.path.join(dataset_dir, 'original_images')
    output_folder = os.path.join(dataset_dir, 'cropped_images')
    viz_output_folder = os.path.join(dataset_dir, 'grid_images')

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(viz_output_folder, exist_ok=True)

    # Calculate effective crop size with overlap
    effective_crop_size = crop_size - overlap

    # Walk through all files in input folder
    image_files = []
    for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.TIF', '.TIFF', '.JPG', '.JPEG', '.PNG']:
        image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])

    print(f"Processing files in {input_folder}: {len(image_files)} images found")

    for img_file in image_files:
        try:
            # Read image
            img_path = os.path.join(input_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            # Get image dimensions
            height, width = img.shape[:2]
            print(f"Processing image: {img_file}, dimensions: {width}x{height}")

            # For high-resolution images, adjust grid size if needed
            actual_grid_size = grid_size
            if width >= 3000 or height >= 2000:
                # For large microscopy images, we might want to create more crops
                print(f"Large image detected ({width}x{height}), using grid size: {actual_grid_size}")

            # Calculate the total grid area size (with overlap adjustment)
            total_grid_width = effective_crop_size * actual_grid_size + overlap
            total_grid_height = effective_crop_size * actual_grid_size + overlap

            # Calculate starting position to center the grid
            start_x = max(0, (width - total_grid_width) // 2)
            start_y = max(0, (height - total_grid_height) // 2)

            # Adjust if the grid would go beyond image boundaries
            if start_x + total_grid_width > width:
                start_x = max(0, width - total_grid_width)
            if start_y + total_grid_height > height:
                start_y = max(0, height - total_grid_height)

            # Make a copy of the original image for visualization
            if visualize:
                viz_img = img.copy()
                # Draw the overall grid boundary in red
                cv2.rectangle(viz_img,
                             (start_x, start_y),
                             (start_x + total_grid_width, start_y + total_grid_height),
                             (0, 0, 255), 3)  # Red, thickness 3

            # Create base filename for crops
            base_name = os.path.splitext(img_file)[0]

            # Process each grid cell
            crop_count = 0
            for row in range(actual_grid_size):
                for col in range(actual_grid_size):
                    # Skip the 4 corner cells
                    if (row == 0 and col == 0) or \
                       (row == 0 and col == actual_grid_size - 1) or \
                       (row == actual_grid_size - 1 and col == 0) or \
                       (row == actual_grid_size - 1 and col == actual_grid_size - 1):
                        continue

                    # Calculate crop coordinates (with overlap adjustment)
                    x = start_x + col * effective_crop_size
                    y = start_y + row * effective_crop_size

                    # Make sure crop is within image bounds
                    if x + crop_size > width or y + crop_size > height:
                        print(f"  Skipping crop at r{row}c{col} - would extend beyond image boundaries")
                        continue

                    # Extract the crop
                    crop = img[y:y+crop_size, x:x+crop_size]

                    # Skip empty or nearly empty crops (optional)
                    if np.mean(crop) < 10:  # Skip very dark crops
                        print(f"  Skipping dark crop at r{row}c{col}")
                        continue

                    # Save crop
                    crop_name = f"{base_name}_grid_r{row}_c{col}.png"
                    crop_path = os.path.join(output_folder, crop_name)
                    cv2.imwrite(crop_path, crop)
                    crop_count += 1

                    # Draw rectangle for this crop on visualization image
                    if visualize:
                        color = (0, 255, 0)  # Green for normal grid cells
                        cv2.rectangle(viz_img,
                                     (x, y),
                                     (x + crop_size, y + crop_size),
                                     color, 2)

                        # Add grid position text
                        text = f"r{row}c{col}"
                        cv2.putText(viz_img, text,
                                   (x + 5, y + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                   (255, 255, 255), 2)

            print(f"Created {crop_count} crops from {img_path}")

            # Draw the corner cells that are skipped (in black)
            if visualize:
                corner_positions = [
                    (0, 0),  # Top-left
                    (0, actual_grid_size - 1),  # Top-right
                    (actual_grid_size - 1, 0),  # Bottom-left
                    (actual_grid_size - 1, actual_grid_size - 1)  # Bottom-right
                ]

                for row, col in corner_positions:
                    x = start_x + col * effective_crop_size
                    y = start_y + row * effective_crop_size
                    cv2.rectangle(viz_img,
                                 (x, y),
                                 (x + crop_size, y + crop_size),
                                 (0, 0, 0), 2)  # Black for skipped corners

                # Save visualization image to a separate folder
                viz_name = f"{base_name}_grid_visualization.jpg"
                viz_path = os.path.join(viz_output_folder, viz_name)
                cv2.imwrite(viz_path, viz_img)
                print(f"Created visualization image: {viz_path}")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            import traceback
            traceback.print_exc()

def match_cropped_images_with_density(dataset_dir):
    """
    Create a CSV file with cropped image filenames and their corresponding density values.
    """
    # Set up paths
    cropped_images_path = os.path.join(dataset_dir, 'cropped_images')
    density_with_filenames_csv = os.path.join(dataset_dir, 'beadDensity_with_filenames.csv')
    output_csv_path = os.path.join(dataset_dir, 'cropped_images_with_density.csv')

    # Create directory if it doesn't exist
    os.makedirs(cropped_images_path, exist_ok=True)

    # Read the previously created CSV with original filenames and densities
    try:
        original_df = pd.read_csv(density_with_filenames_csv)
    except Exception as e:
        print(f"Error reading density CSV: {e}")
        print("Creating an empty CSV file with proper structure")
        original_df = pd.DataFrame(columns=['filename', 'density'])

    # Create a dictionary mapping original filenames to density values
    # Remove file extensions for proper matching
    filename_to_density = {}
    for _, row in original_df.iterrows():
        # Extract the base part of the filename (without extension)
        base_name = os.path.splitext(row['filename'])[0] if '.' in str(row['filename']) else str(row['filename'])
        density = row['density']
        filename_to_density[base_name] = density

    # Get list of cropped image filenames
    cropped_files = []
    for file in os.listdir(cropped_images_path):
        if file.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            cropped_files.append(file)

    print(f"Found {len(cropped_files)} cropped images")

    # Create a list to store results
    results = []

    # Process each cropped image
    for cropped_file in cropped_files:
        # Remove extension for processing
        cropped_base = os.path.splitext(cropped_file)[0]

        # Extract the original filename part (before _grid)
        match = re.match(r'(.*?)_grid_r\d+_c\d+$', cropped_base)

        if match:
            original_base = match.group(1)

            # Find corresponding density
            density = None

            # First try direct matching with the base name
            if original_base in filename_to_density:
                density = filename_to_density[original_base]
            else:
                # Try matching with different patterns in the original filenames
                for orig_name in filename_to_density:
                    # Try matching with the beginning of the filename
                    if orig_name.startswith(original_base) or original_base.startswith(orig_name):
                        density = filename_to_density[orig_name]
                        break

                    # Try extracting patterns like date or time
                    date_match = re.search(r'(\d{2}-\d{2}-\d{2}[^_]+)', original_base)
                    if date_match and date_match.group(1) in orig_name:
                        density = filename_to_density[orig_name]
                        break

            # Add to results
            results.append({'filename': cropped_base, 'density': density})
        else:
            # If no grid pattern found, add with unknown density
            results.append({'filename': cropped_base, 'density': None})

    # Create DataFrame from results
    result_df = pd.DataFrame(results)

    # Sort the DataFrame by the grid position (row, then column)
    result_df['row'] = result_df['filename'].apply(
        lambda x: int(re.search(r'_grid_r(\d+)_c\d+', x).group(1)) if re.search(r'_grid_r(\d+)_c\d+', x) else 999)

    result_df['col'] = result_df['filename'].apply(
        lambda x: int(re.search(r'_grid_r\d+_c(\d+)', x).group(1)) if re.search(r'_grid_r\d+_c(\d+)', x) else 999)

    # Extract base image name (without grid position)
    result_df['base_image'] = result_df['filename'].apply(
        lambda x: re.match(r'(.*?)_grid_r\d+_c\d+', x).group(1) if re.match(r'(.*?)_grid_r\d+_c\d+', x) else x)

    # Sort by base image name, then row, then column
    result_df = result_df.sort_values(['base_image', 'row', 'col'])

    # Drop the sorting columns before saving
    result_df = result_df.drop(['row', 'col', 'base_image'], axis=1)

    # Save to CSV
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"New CSV file created at {output_csv_path}")

    # Also create a simple density.csv in the format expected by the training script
    simple_csv_path = os.path.join(dataset_dir, 'density.csv')
    result_df.to_csv(simple_csv_path, index=False, header=False, encoding='utf-8-sig')
    print(f"Simple density CSV created at {simple_csv_path}")

    return result_df

def main():
    """
    Main function to run the entire workflow.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Process microscopy images for CNN training')
    parser.add_argument('--input_dir', type=str, default=None,
                      help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save processed images')
    parser.add_argument('--density_csv', type=str, default=None,
                      help='CSV file with density values')
    parser.add_argument('--grid_cols', type=int, default=8,
                      help='Number of grid columns (default: 8 for 3840px width)')
    parser.add_argument('--grid_rows', type=int, default=4,
                      help='Number of grid rows (default: 4 for 2160px height)')
    parser.add_argument('--crop_size', type=int, default=512,
                      help='Size of each crop in pixels (default: 512)')
    parser.add_argument('--no_visualize', action='store_false', dest='visualize',
                      help='Disable visualization image creation')
    parser.add_argument('--skip_organize', action='store_true',
                      help='Skip organizing images step')
    args = parser.parse_args()

    # Set up paths
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # Step 1: Organize images and create initial CSV
    if not args.skip_organize:
        print("\n== Step 1: Organizing Images ==")
        organize_images(args.input_dir, args.output_dir, args.density_csv)

    # Step 2: Crop images and create grid visualizations
    print("\n== Step 2: Cropping Images ==")
    crop_center_grid(dataset_dir, args.grid_cols, args.grid_rows, args.crop_size, args.visualize)

    # Step 3: Match cropped images with density values
    print("\n== Step 3: Matching Cropped Images with Density Values ==")
    match_cropped_images_with_density(dataset_dir)

    print("\nComplete workflow finished successfully!")

if __name__ == "__main__":
    main()
