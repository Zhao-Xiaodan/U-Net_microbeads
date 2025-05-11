
#!/usr/bin/env python3
"""
Enhanced implementation for bead instance segmentation using three-class approach
Classes: Background (0), Bead Interior (1), Bead Boundaries (2)
V5: Improved clustering separation with enhanced preprocessing and watershed algorithm
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature, morphology, measure, filters
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from glob import glob
import gc
import traceback
import argparse

# Enable memory cleanup
gc.enable()

# Force CPU-only mode for stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')  # More aggressive CPU-only setting

# Limit TensorFlow memory growth
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Limit the number of threads for better stability
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Disable Intel optimizations that might cause issues on Apple Silicon
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = 'shared'

# Clear any existing models/sessions
tf.keras.backend.clear_session()
gc.collect()

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")
CIRCULAR_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "circular_masks")
INSTANCE_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "instance_masks")
BOUNDARY_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "boundary_masks")
THREE_CLASS_MASK_FOLDER = os.path.join(BASE_DIR, "dataset", "three_class_masks")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

# Ensure output folders exist
for folder in [RESULTS_FOLDER, MODELS_FOLDER, THREE_CLASS_MASK_FOLDER]:
    os.makedirs(folder, exist_ok=True)

#####################################
# DATA PREPARATION FUNCTIONS        #
#####################################

def preprocess_image(img, disk_size=50):
    """
    Advanced preprocessing for microscopy images with improved features for cluster separation

    Args:
        img: Input grayscale image (0-1 float)
        disk_size: Size of structuring element for background extraction

    Returns:
        Preprocessed image (0-1 float with 3 channels)
    """
    try:
        # Convert to 0-255 for opencv operations
        img_uint8 = (img * 255).astype(np.uint8)

        # Create a structuring element for background estimation
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))

        # Background estimation using morphological opening
        background = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, selem)

        # Subtract background and add back mean to preserve brightness
        mean_bg = np.mean(background)
        corrected = img_uint8.astype(float) - background.astype(float) + mean_bg
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(corrected)

        # Add Laplacian of Gaussian to highlight circular structures
        log_kernel_size = 5
        log = cv2.GaussianBlur(img_uint8, (log_kernel_size, log_kernel_size), 0)
        log = cv2.Laplacian(log, cv2.CV_64F)
        log = (log - np.min(log)) / (np.max(log) - np.min(log) + 1e-8)  # Normalize to 0-1

        # Create binary mask for distance transform
        # Use Otsu's thresholding to get a binary mask
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Apply morphological operations to clean up the binary mask
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Apply distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        # Return multi-channel result with enhanced features for separation
        return np.stack([enhanced / 255.0, log, dist_normalized], axis=-1)
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        # Return original image with placeholder channels
        return np.stack([img, np.zeros_like(img), np.zeros_like(img)], axis=-1)

def generate_three_class_masks(instance_masks, save_dir=None):
    """
    Generate three-class masks from instance masks with improved boundary detection:
    0: Background
    1: Bead Interior
    2: Bead Boundaries

    Args:
        instance_masks: List of file paths or array of instance masks
        save_dir: Directory to save generated masks (optional)

    Returns:
        Three-class masks array if instance_masks is array, otherwise None
    """
    three_class_masks = []

    print(f"Generating three-class masks from instance masks...")

    if isinstance(instance_masks, list):
        # List of file paths
        for mask_path in instance_masks:
            try:
                # Load instance mask
                instance_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if instance_mask is None:
                    print(f"Error loading mask: {mask_path}")
                    continue

                # Generate three-class mask
                three_class = create_three_class_mask(instance_mask)

                # Save if directory provided
                if save_dir:
                    filename = os.path.basename(mask_path)
                    output_path = os.path.join(save_dir, filename)
                    cv2.imwrite(output_path, three_class)
                    print(f"Saved three-class mask to {output_path}")

            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
                continue

    else:
        # Array of masks
        for i, instance_mask in enumerate(instance_masks):
            try:
                # Generate three-class mask
                three_class = create_three_class_mask(instance_mask)
                three_class_masks.append(three_class)

                # Save if directory provided
                if save_dir:
                    output_path = os.path.join(save_dir, f"mask_{i:04d}.png")
                    cv2.imwrite(output_path, three_class)

                if i % 10 == 0:
                    print(f"Processed {i+1}/{len(instance_masks)} masks")

            except Exception as e:
                print(f"Error processing mask {i}: {e}")
                # Add empty mask as placeholder
                three_class_masks.append(np.zeros_like(instance_mask))

        return np.array(three_class_masks)

def create_three_class_mask(instance_mask):
    """
    Create a three-class mask from a single instance mask with enhanced boundary detection

    Args:
        instance_mask: Instance segmentation mask with unique IDs

    Returns:
        Three-class mask: Background (0), Interior (1), Boundary (2)
    """
    # Initialize three-class mask (all background)
    three_class = np.zeros_like(instance_mask, dtype=np.uint8)

    # Get unique instance IDs (excluding background 0)
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]

    # Set all foreground to interior class (1)
    three_class[instance_mask > 0] = 1

    # For each instance, find and set boundary to class 2
    boundaries = np.zeros_like(instance_mask, dtype=bool)

    for instance_id in unique_ids:
        # Create binary mask for this instance
        instance_binary = (instance_mask == instance_id)

        # Find boundary pixels using more aggressive erosion
        # Use a slightly larger structuring element for more prominent boundaries
        selem = morphology.disk(2)  # Increase from 1 to 2 for thicker boundaries
        eroded = morphology.binary_erosion(instance_binary, selem)
        instance_boundary = instance_binary & ~eroded

        # Add to combined boundary mask
        boundaries |= instance_boundary

    # Set boundaries in three-class mask (class 2)
    three_class[boundaries] = 2

    return three_class

def prepare_three_class_dataset():
    """
    Prepare three-class masks dataset from existing instance masks
    """
    print("Preparing three-class masks dataset...")

    # Check if instance masks exist
    instance_mask_files = glob(os.path.join(INSTANCE_MASK_FOLDER, "*.png"))

    if len(instance_mask_files) == 0:
        print(f"ERROR: No instance masks found in {INSTANCE_MASK_FOLDER}")
        return False

    print(f"Found {len(instance_mask_files)} instance masks")

    # Generate three-class masks
    generate_three_class_masks(instance_mask_files, save_dir=THREE_CLASS_MASK_FOLDER)

    # Verify creation
    three_class_files = glob(os.path.join(THREE_CLASS_MASK_FOLDER, "*.png"))
    print(f"Generated {len(three_class_files)} three-class masks")

    if len(three_class_files) > 0:
        # Visualize a few samples
        visualize_three_class_masks(three_class_files[:5])
        return True
    else:
        print("Failed to generate three-class masks")
        return False

def visualize_three_class_masks(mask_files):
    """
    Visualize three-class masks to verify correct generation

    Args:
        mask_files: List of mask file paths to visualize
    """
    plt.figure(figsize=(15, 5 * len(mask_files)))

    for i, mask_path in enumerate(mask_files):
        try:
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if mask is None:
                print(f"Error loading mask: {mask_path}")
                continue

            # For visualization, create a color map
            colors = np.array([
                [0, 0, 0],        # Background (black)
                [0, 0, 255],      # Interior (blue)
                [255, 0, 0]       # Boundary (red)
            ], dtype=np.uint8)

            # Create RGB visualization
            rgb_mask = colors[mask]

            # Display
            plt.subplot(len(mask_files), 2, 2*i + 1)
            plt.imshow(mask, cmap='viridis')
            plt.title(f"Three-class mask {i+1}")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(len(mask_files), 2, 2*i + 2)
            plt.imshow(rgb_mask)
            plt.title(f"Visualization (Blue=Interior, Red=Boundary)")
            plt.axis('off')

        except Exception as e:
            print(f"Error visualizing mask {mask_path}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "three_class_mask_visualization.png"))
    plt.close()

def load_images_from_directory(image_dir, mask_dir, target_size=(256, 256), mask_type="three_class", max_images=None):
    """
    Load images and corresponding masks with optional type selection

    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing masks
        target_size: Size to resize images to
        mask_type: Type of mask to use - 'three_class', 'instance', 'binary', etc.
        max_images: Maximum number of images to load (for testing with small datasets)
    """
    images, masks = [], []

    try:
        image_paths = glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.png'))

        # Limit number of images if specified
        if max_images is not None and max_images > 0:
            image_paths = image_paths[:max_images]

        print(f"Found {len(image_paths)} images")

        # Select the appropriate mask folder based on mask_type
        if mask_type == "binary":
            mask_dir = MASK_FOLDER
        elif mask_type == "circular":
            mask_dir = CIRCULAR_MASK_FOLDER
        elif mask_type == "boundary":
            mask_dir = BOUNDARY_MASK_FOLDER
        elif mask_type == "instance":
            mask_dir = INSTANCE_MASK_FOLDER
        elif mask_type == "three_class":
            mask_dir = THREE_CLASS_MASK_FOLDER

        print(f"Using masks from: {mask_dir}")

        for image_path in image_paths:
            try:
                # Load image
                img = load_img(image_path, color_mode="grayscale", target_size=target_size)
                img = img_to_array(img) / 255.0

                # Apply enhanced preprocessing (now with 3 channels)
                processed_img = preprocess_image(img.squeeze())

                # Get corresponding mask filename
                image_name = os.path.basename(image_path)
                mask_path = os.path.join(mask_dir, image_name)

                # For instance or three-class masks, use PNG extension
                if mask_type in ["instance", "three_class"]:
                    mask_path = os.path.splitext(mask_path)[0] + ".png"

                if os.path.exists(mask_path):
                    # Load mask
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        print(f"Warning: Failed to load mask: {mask_path}")
                        continue

                    # Resize to target size
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

                    # Process mask based on type
                    if mask_type == "three_class":
                        # For three-class masks, convert to categorical format for training
                        mask_categorical = tf.keras.utils.to_categorical(mask, num_classes=3)
                        mask = mask_categorical
                    elif mask_type == "instance":
                        # For instance masks, preserve instance IDs
                        mask = mask.astype(np.float32)
                        mask = np.expand_dims(mask, axis=-1)
                    else:
                        # For binary masks, convert to 0-1
                        mask = (mask > 127).astype(np.float32)
                        mask = np.expand_dims(mask, axis=-1)

                    images.append(processed_img)
                    masks.append(mask)

                    print(f"Processed {image_name}")
                else:
                    print(f"Mask not found for image {image_name}")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        # Stack into arrays
        if images:
            return np.array(images), np.array(masks)
        else:
            print("No valid images loaded!")
            return np.array([]), np.array([])

    except Exception as e:
        print(f"Error in load_images_from_directory: {e}")
        return np.array([]), np.array([])

def convert_instance_to_three_class_batch(instance_masks):
    """
    Convert a batch of instance masks to three-class masks

    Args:
        instance_masks: Array of instance masks (batch_size, H, W, 1)

    Returns:
        Three-class masks (batch_size, H, W, 3) in categorical format
    """
    three_class_masks = []

    for mask in instance_masks:
        # Create three-class mask with enhanced boundaries
        three_class = create_three_class_mask(mask.squeeze())

        # Convert to categorical
        three_class_categorical = tf.keras.utils.to_categorical(three_class, num_classes=3)

        three_class_masks.append(three_class_categorical)

    return np.array(three_class_masks)

#####################################
# MODEL DEFINITION                  #
#####################################

# Custom layer for normalization
class NormalizationLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, training=None):
        return (inputs - tf.reduce_mean(inputs)) / (tf.math.reduce_std(inputs) + 1e-8)

    def get_config(self):
        return super(NormalizationLayer, self).get_config()

# Define losses and metrics for multi-class segmentation
def categorical_dice_loss(y_true, y_pred, smooth=1.0):
    """
    Categorical Dice loss for multi-class segmentation with enhanced boundary weight

    Args:
        y_true: One-hot encoded ground truth
        y_pred: Predicted probabilities
        smooth: Smoothing factor

    Returns:
        Average Dice loss across all classes with higher weight for boundaries
    """
    # Number of classes
    num_classes = y_pred.shape[-1]

    # Ensure both tensors are same type (float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Initialize losses
    dice_losses = []

    # Calculate dice loss for each class
    for class_idx in range(num_classes):
        # Extract class
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]

        # Flatten
        y_true_flat = tf.reshape(y_true_class, [-1])
        y_pred_flat = tf.reshape(y_pred_class, [-1])

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        true_sum = tf.reduce_sum(y_true_flat)
        pred_sum = tf.reduce_sum(y_pred_flat)

        # Calculate Dice coefficient
        dice = (2. * intersection + smooth) / (true_sum + pred_sum + smooth)
        dice_losses.append(1 - dice)

    # Increased weight for boundary class to improve separation
    weights = [0.25, 0.25, 0.5]  # Background, Interior, Boundary
    weighted_loss = sum(w * l for w, l in zip(weights, dice_losses))

    return weighted_loss

def weighted_categorical_crossentropy(weights=[0.5, 0.5, 3.0]):
    """
    Weighted categorical crossentropy with increased weight for boundaries

    Args:
        weights: Class weights (increased weight for boundaries)

    Returns:
        Weighted loss function
    """
    weights_tensor = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Ensure consistent types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Scale predictions
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Weighted crossentropy
        loss = y_true * tf.math.log(y_pred) * weights_tensor
        loss = -tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)

    return loss

def mean_iou(y_true, y_pred):
    """
    Mean IoU metric for multi-class segmentation

    Args:
        y_true: One-hot encoded ground truth
        y_pred: Predicted probabilities

    Returns:
        Mean IoU across classes
    """
    # Ensure consistent types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Convert predictions to class indices
    y_pred_classes = tf.argmax(y_pred, axis=-1)
    y_pred_one_hot = tf.one_hot(y_pred_classes, depth=y_pred.shape[-1], dtype=tf.float32)

    # Calculate IoU for each class
    ious = []
    for class_idx in range(y_pred.shape[-1]):
        # Extract class
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred_one_hot[..., class_idx]

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection

        # Calculate IoU
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)

    # Return mean IoU
    return tf.reduce_mean(ious)

def boundary_accuracy(y_true, y_pred):
    """
    Accuracy specifically for the boundary class

    Args:
        y_true: One-hot encoded ground truth
        y_pred: Predicted probabilities

    Returns:
        Accuracy for boundary class
    """
    # Ensure consistent types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Convert predictions to class indices
    y_pred_classes = tf.argmax(y_pred, axis=-1)

    # Extract ground truth for boundary class (class 2)
    y_true_boundary = y_true[..., 2]

    # True positives for boundary
    boundary_correct = tf.cast(tf.equal(y_pred_classes, 2), tf.float32) * y_true_boundary

    # Calculate accuracy
    boundary_total = tf.reduce_sum(y_true_boundary) + tf.keras.backend.epsilon()
    accuracy = tf.reduce_sum(boundary_correct) / boundary_total

    return accuracy

# Define combined loss function with higher boundary weight
def combined_loss(y_true, y_pred):
    """
    Combined loss function using both Dice and weighted cross-entropy
    with increased emphasis on boundaries

    Args:
        y_true: One-hot encoded ground truth
        y_pred: Predicted probabilities

    Returns:
        Combined loss value
    """
    # Ensure consistent types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate individual losses
    dice = categorical_dice_loss(y_true, y_pred)
    wce = weighted_categorical_crossentropy(weights=[0.5, 0.5, 3.0])(y_true, y_pred)

    # Return weighted combination
    return 0.5 * dice + 0.5 * wce

# Build an enhanced U-Net for three-class segmentation
def build_three_class_unet(input_shape=(256, 256, 3), filters_base=32):
    """
    Build an enhanced U-Net model for three-class segmentation with improved
    architecture for handling clusters:
    - Background (0)
    - Bead Interior (1)
    - Bead Boundaries (2)

    Args:
        input_shape: Input tensor shape (H, W, C) - now with 3 channels
        filters_base: Base number of filters
    """
    try:
        inputs = layers.Input(shape=input_shape)

        # Normalization
        x = NormalizationLayer()(inputs)

        # Encoder (downsampling path)
        # Level 1
        conv1 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(x)
        conv1 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        # Level 2
        conv2 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Level 3
        conv3 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # Level 4
        conv4 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(conv4)
        drop4 = layers.Dropout(0.2)(conv4)  # Add dropout for regularization
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bridge
        conv5 = layers.Conv2D(filters_base*16, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(filters_base*16, 3, activation='relu', padding='same')(conv5)
        drop5 = layers.Dropout(0.2)(conv5)  # Add dropout

        # Decoder (upsampling path)
        # Level 4
        up6 = layers.UpSampling2D(size=(2, 2))(drop5)
        up6 = layers.Conv2D(filters_base*8, 2, activation='relu', padding='same')(up6)
        merge6 = layers.concatenate([drop4, up6], axis=3)
        conv6 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(conv6)

        # Level 3
        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = layers.Conv2D(filters_base*4, 2, activation='relu', padding='same')(up7)
        merge7 = layers.concatenate([conv3, up7], axis=3)
        conv7 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(conv7)

        # Level 2
        up8 = layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = layers.Conv2D(filters_base*2, 2, activation='relu', padding='same')(up8)
        merge8 = layers.concatenate([conv2, up8], axis=3)
        conv8 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(merge8)
        conv8 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(conv8)

        # Level 1
        up9 = layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = layers.Conv2D(filters_base, 2, activation='relu', padding='same')(up9)
        merge9 = layers.concatenate([conv1, up9], axis=3)
        conv9 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(merge9)
        conv9 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(conv9)

        # Add attention gate for better focus on boundaries
        # Simple self-attention mechanism
        attention = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        conv9 = layers.multiply([conv9, attention])

        # Output layer (3 classes - softmax activation)
        outputs = layers.Conv2D(3, 1, activation='softmax', name='predictions')(conv9)

        # Create and compile model
        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model with enhanced loss and boundary focus
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=combined_loss,
            metrics=['accuracy', mean_iou, boundary_accuracy]
        )

        return model
    except Exception as e:
        print(f"Error in build_three_class_unet: {e}")
        traceback.print_exc()
        return None

#####################################
# IMPROVED TRAINING LOOP           #
#####################################

def train_three_class_model(model, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=2):
    """
    Train the three-class segmentation model with careful training loop

    Args:
        model: The compiled model to train
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        epochs: Number of epochs to train
        batch_size: Batch size for training

    Returns:
        history: Dictionary containing training history
    """
    # Set up optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Training history tracking
    history = {
        'loss': [],
        'val_loss': [],
        'val_mean_iou': [],
        'val_boundary_accuracy': []
    }

    # Best model tracking
    best_val_iou = 0
    best_epoch = 0

    try:
        print("Starting careful training loop for three-class model...")

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            batch_count = 0

            # Shuffle indices
            indices = np.random.permutation(len(X_train))

            # Process each batch
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:min(i+batch_size, len(indices))]
                X_batch = X_train[batch_indices]
                Y_batch = Y_train[batch_indices]

                print(f"  Processing batch {i//batch_size + 1}/{(len(indices)-1)//batch_size + 1}")

                # Clear session periodically to avoid memory leaks
                if (i//batch_size) % 10 == 0 and i > 0:
                    print("  Clearing session to free memory...")
                    gc.collect()

                # Manual training step with gradient tape
                with tf.GradientTape() as tape:
                    # Forward pass
                    print("  Forward pass...")
                    predictions = model(X_batch, training=True)

                    # Compute loss (using the model's loss function)
                    print("  Computing loss...")
                    loss = model.loss(Y_batch, predictions)

                    print(f"  Loss: {loss.numpy():.4f}")

                # Compute gradients
                print("  Computing gradients...")
                gradients = tape.gradient(loss, model.trainable_variables)

                # Apply gradients
                print("  Applying gradients...")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Update metrics
                epoch_loss += loss.numpy()
                batch_count += 1

                # Force memory cleanup
                gc.collect()

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count

            # Update history
            history['loss'].append(avg_epoch_loss)

            # Validate with single sample prediction for stability
            print("Evaluating on validation data...")
            val_losses = []
            val_mean_ious = []
            val_boundary_accs = []

            for i in range(0, len(X_val), batch_size):
                end = min(i + batch_size, len(X_val))
                X_batch_val = X_val[i:end]
                Y_batch_val = Y_val[i:end]

                print(f"  Processing validation batch {i//batch_size + 1}/{(len(X_val)-1)//batch_size + 1}")

                # Process validation samples individually for stability
                for j in range(len(X_batch_val)):
                    X_sample = np.expand_dims(X_batch_val[j], axis=0)
                    Y_sample = np.expand_dims(Y_batch_val[j], axis=0)

                    # Prediction
                    pred = model(X_sample, training=False)

                    # Calculate validation metrics
                    val_loss = model.loss(Y_sample, pred).numpy()
                    val_losses.append(val_loss)

                    # Calculate mean IoU
                    iou = mean_iou(Y_sample, pred).numpy()
                    val_mean_ious.append(iou)

                    # Calculate boundary accuracy
                    b_acc = boundary_accuracy(Y_sample, pred).numpy()
                    val_boundary_accs.append(b_acc)

                # Memory cleanup
                gc.collect()

            # Average validation metrics
            avg_val_loss = np.mean(val_losses)
            avg_val_iou = np.mean(val_mean_ious)
            avg_val_boundary_acc = np.mean(val_boundary_accs)

            # Update history
            history['val_loss'].append(avg_val_loss)
            history['val_mean_iou'].append(avg_val_iou)
            history['val_boundary_accuracy'].append(avg_val_boundary_acc)

            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f} - val_loss: {avg_val_loss:.4f} - val_iou: {avg_val_iou:.4f} - val_boundary_acc: {avg_val_boundary_acc:.4f}")

            # Save best model
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                best_epoch = epoch

                try:
                    print("Saving best model...")
                    model.save(os.path.join(MODELS_FOLDER, "best_three_class_model.keras"))
                    print(f"Model saved to {os.path.join(MODELS_FOLDER, 'best_three_class_model.keras')}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                    traceback.print_exc()

            # Memory cleanup
            gc.collect()

        print(f"Training completed. Best model from epoch {best_epoch+1} with IoU: {best_val_iou:.4f}")
        return history

    except Exception as e:
        print(f"Error in training loop: {e}")
        traceback.print_exc()
        return history

def plot_training_history(history):
    """
    Plot training history for the three-class model

    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(15, 10))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # IoU plot
    plt.subplot(2, 2, 3)
    plt.plot(history['val_mean_iou'], label='Validation Mean IoU')
    plt.title('Mean IoU History')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    # Boundary accuracy plot
    plt.subplot(2, 2, 4)
    plt.plot(history['val_boundary_accuracy'], label='Boundary Accuracy')
    plt.title('Boundary Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "three_class_training_history.png"))
    plt.close()

#####################################
# PREDICTION AND POST-PROCESSING    #
#####################################

def load_model_safely(model_path):
    """
    Load a model with enhanced safety measures to prevent segmentation faults

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded model or None if loading fails
    """
    print(f"Loading model from {model_path}...")

    # First, clear any existing models/sessions
    tf.keras.backend.clear_session()
    gc.collect()

    try:
        # Try to load the model without compiling first
        print("Loading model without compiling...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'NormalizationLayer': NormalizationLayer,
                'categorical_dice_loss': categorical_dice_loss,
                'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
                'combined_loss': combined_loss,
                'mean_iou': mean_iou,
                'boundary_accuracy': boundary_accuracy
            },
            compile=False
        )

        # Compile after loading
        print("Compiling model...")
        model.compile(
            optimizer='adam',  # Use a simpler optimizer
            loss=combined_loss,
            metrics=['accuracy', mean_iou, boundary_accuracy]
        )

        print("Successfully loaded and compiled model")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def predict_three_class(model, images, batch_size=1):
    """
    Generate three-class predictions with enhanced safety measures

    Args:
        model: Loaded model
        images: Images to predict
        batch_size: Batch size for prediction (use 1 for maximum stability)

    Returns:
        Three-class predictions (softmax probabilities)
    """
    predictions = []

    try:
        print(f"Generating predictions for {len(images)} images...")

        # Process each image individually for maximum stability
        for i, img in enumerate(images):
            try:
                print(f"  Predicting image {i+1}/{len(images)}...")

                # Create a batch of size 1
                img_batch = np.expand_dims(img, axis=0)

                # Generate prediction
                pred = model.predict(img_batch, verbose=0)
                predictions.append(pred[0])

                # Force memory cleanup
                if (i+1) % 10 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Error predicting image {i+1}: {e}")
                # Add an empty prediction as a placeholder
                empty_pred = np.zeros((*img.shape[:-1], 3))
                predictions.append(empty_pred)
                continue

        return np.array(predictions)

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        traceback.print_exc()
        return np.array(predictions)

def extract_instances_from_three_class(three_class_pred, min_size=30):
    """
    Extract instance masks from three-class predictions with improved
    separation of clustered beads

    Args:
        three_class_pred: Three-class prediction (H, W, 3)
        min_size: Minimum size of objects to keep

    Returns:
        Instance mask with unique IDs for each instance
    """
    try:
        # Convert softmax probabilities to class indices
        class_indices = np.argmax(three_class_pred, axis=-1)

        # Extract binary mask (interior + boundary)
        binary_mask = (class_indices > 0).astype(np.uint8)

        # Extract boundary mask
        boundary_mask = (class_indices == 2).astype(np.uint8)

        # Extract interior mask (class 1)
        interior_mask = (class_indices == 1).astype(np.uint8)

        # Use boundary as markers to separate touching objects
        # Invert boundary to create a mask where boundaries are 0
        boundary_inverse = 1 - boundary_mask

        # Apply boundary mask to binary mask to create separated objects
        separated_objects = binary_mask * boundary_inverse

        # Calculate distance transform on interior for better seeding
        distance = ndi.distance_transform_edt(interior_mask)

        # Find local maxima to use as markers for watershed
        # This is key for separating clustered beads
        peaks = peak_local_max(
            distance,
            min_distance=5,  # Adjusted to better separate close beads
            threshold_abs=0.2,  # Low threshold to find more peaks
            indices=False,
            labels=interior_mask
        )

        # Label peaks as initial markers
        markers = ndi.label(peaks)[0]

        # Apply watershed for instance segmentation
        watershed_result = watershed(-distance, markers, mask=binary_mask)

        # Clean up small objects
        for label in np.unique(watershed_result):
            if label == 0:  # Skip background
                continue

            # Get object size
            size = np.sum(watershed_result == label)

            # Remove if smaller than minimum size
            if size < min_size:
                watershed_result[watershed_result == label] = 0

        # Relabel to ensure consecutive labels
        watershed_result = measure.label(watershed_result > 0)

        return watershed_result

    except Exception as e:
        print(f"Error in extract_instances_from_three_class: {e}")
        traceback.print_exc()
        return np.zeros_like(three_class_pred[..., 0], dtype=np.int32)

def process_predictions_to_instances(three_class_predictions, min_size=30):
    """
    Process batch of three-class predictions to instance masks

    Args:
        three_class_predictions: Batch of three-class predictions
        min_size: Minimum size of objects to keep

    Returns:
        Batch of instance masks
    """
    instance_masks = []

    for i, pred in enumerate(three_class_predictions):
        try:
            print(f"Processing prediction {i+1}/{len(three_class_predictions)}...")

            # Extract instances from three-class prediction
            instances = extract_instances_from_three_class(pred, min_size=min_size)

            # Add to result
            instance_masks.append(instances)

            # Memory cleanup
            if (i+1) % 10 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error processing prediction {i+1}: {e}")
            # Add empty mask as placeholder
            instance_masks.append(np.zeros(pred.shape[:-1], dtype=np.int32))

    return np.array(instance_masks)

def visualize_three_class_predictions(images, three_class_preds, instance_preds, true_instances=None, max_samples_per_file=5):
    """
    Visualize three-class predictions and extracted instances, creating multiple output files
    with max_samples_per_file images in each file.

    Args:
        images: Original images
        three_class_preds: Three-class predictions
        instance_preds: Extracted instance masks
        true_instances: Ground truth instance masks (optional)
        max_samples_per_file: Maximum number of samples to visualize per output file
    """
    num_samples = len(images)
    num_files = (num_samples + max_samples_per_file - 1) // max_samples_per_file  # Ceiling division

    print(f"Generating {num_files} visualization files for {num_samples} images...")

    # Define colors for visualization
    colors = np.array([
        [0, 0, 0],        # Background (black)
        [0, 0, 255],      # Interior (blue)
        [255, 0, 0]       # Boundary (red)
    ], dtype=np.uint8)

    # Number of columns in the plot (depends on whether true_instances is provided)
    n_cols = 4 if true_instances is not None else 3

    for file_idx in range(num_files):
        start_idx = file_idx * max_samples_per_file
        end_idx = min(start_idx + max_samples_per_file, num_samples)
        samples_in_file = end_idx - start_idx

        print(f"Creating visualization file {file_idx+1}/{num_files} (images {start_idx+1}-{end_idx})...")

        plt.figure(figsize=(4*n_cols, 4*samples_in_file))

        for i in range(samples_in_file):
            img_idx = start_idx + i
            try:
                # Original image (first channel)
                plt.subplot(samples_in_file, n_cols, i*n_cols + 1)
                plt.imshow(images[img_idx][..., 0], cmap='gray')
                plt.title(f"Original Image {img_idx+1}")
                plt.axis('off')

                # Three-class prediction
                plt.subplot(samples_in_file, n_cols, i*n_cols + 2)
                # Convert softmax to class indices
                class_indices = np.argmax(three_class_preds[img_idx], axis=-1)
                # Create RGB visualization
                rgb_pred = colors[class_indices]
                plt.imshow(rgb_pred)
                plt.title(f"Three-class Prediction {img_idx+1}")
                plt.axis('off')

                # Instance prediction
                plt.subplot(samples_in_file, n_cols, i*n_cols + 3)
                plt.imshow(instance_preds[img_idx], cmap='nipy_spectral')
                unique_instances = len(np.unique(instance_preds[img_idx])) - (1 if 0 in np.unique(instance_preds[img_idx]) else 0)
                plt.title(f"Instance Prediction {img_idx+1}\n({unique_instances} objects)")
                plt.axis('off')

                # Ground truth instance mask (if provided)
                if true_instances is not None:
                    plt.subplot(samples_in_file, n_cols, i*n_cols + 4)
                    plt.imshow(true_instances[img_idx].squeeze(), cmap='nipy_spectral')
                    unique_gt = len(np.unique(true_instances[img_idx])) - (1 if 0 in np.unique(true_instances[img_idx]) else 0)
                    plt.title(f"Ground Truth {img_idx+1}\n({unique_gt} objects)")
                    plt.axis('off')

            except Exception as e:
                print(f"Error visualizing sample {img_idx+1}: {e}")

        plt.tight_layout()
        output_filename = os.path.join(RESULTS_FOLDER, f"three_class_predictions_images_{start_idx+1}-{end_idx}.png")
        plt.savefig(output_filename)
        print(f"Saved visualization to {output_filename}")
        plt.close()

    print(f"Created {num_files} visualization files in {RESULTS_FOLDER}")

def compute_instance_metrics(pred_instances, true_instances):
    """
    Compute metrics for instance segmentation quality

    Args:
        pred_instances: Predicted instance masks
        true_instances: Ground truth instance masks

    Returns:
        Dictionary of metrics
    """
    # Basic counting metrics
    metrics = {}
    tp_total, fp_total, fn_total = 0, 0, 0
    gt_count_total, pred_count_total = 0, 0

    # Process each image
    for i in range(len(true_instances)):
        try:
            # Get counts
            gt_count = len(np.unique(true_instances[i])) - (1 if 0 in np.unique(true_instances[i]) else 0)
            pred_count = len(np.unique(pred_instances[i])) - (1 if 0 in np.unique(pred_instances[i]) else 0)

            gt_count_total += gt_count
            pred_count_total += pred_count

            # Simple count-based metrics (approximate)
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)

            tp_total += tp
            fp_total += fp
            fn_total += fn

        except Exception as e:
            print(f"Error computing metrics for sample {i+1}: {e}")

    # Calculate aggregate metrics
    if tp_total + fp_total > 0:
        precision = tp_total / (tp_total + fp_total)
    else:
        precision = 0

    if tp_total + fn_total > 0:
        recall = tp_total / (tp_total + fn_total)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gt_instances': gt_count_total,
        'pred_instances': pred_count_total
    }

    return metrics

#####################################
# MAIN FUNCTION                     #
#####################################

def main_three_class_approach():
    """
    Main function for three-class approach to instance segmentation with improved
    clustering handling:
    1. Generate three-class masks from instance masks with enhanced boundaries
    2. Train a three-class segmentation model with 3-channel input
    3. Predict and convert to instance masks using enhanced watershed algorithm
    """
    try:
        print("=" * 50)
        print("THREE-CLASS INSTANCE SEGMENTATION WITH CLUSTER SEPARATION")
        print("=" * 50)

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train a three-class instance segmentation model with improved cluster handling')
        parser.add_argument('--max-images', type=int, default=None,
                          help='Maximum number of images to load (for testing)')
        parser.add_argument('--batch-size', type=int, default=2,
                          help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=30,
                          help='Number of epochs to train')
        parser.add_argument('--filters', type=int, default=32,
                          help='Base number of filters in the U-Net')
        parser.add_argument('--image-size', type=int, default=256,
                          help='Image size for training')
        parser.add_argument('--skip-training', action='store_true',
                          help='Skip training and only do prediction/evaluation')
        parser.add_argument('--predict-only', action='store_true',
                          help='Only run prediction using existing model')
        parser.add_argument('--min-object-size', type=int, default=30,
                          help='Minimum object size for instance segmentation')
        parser.add_argument('--prepare-data-only', action='store_true',
                          help='Only prepare three-class dataset and exit')
        parser.add_argument('--visualize', action='store_true', default=True,
                          help='Create visualizations of the results')
        args = parser.parse_args()

        # Step 1: Prepare three-class dataset from instance masks
        print("\nStep 1: Preparing three-class dataset from instance masks...")
        prepare_three_class_dataset()

        if args.prepare_data_only:
            print("Three-class dataset preparation completed. Exiting...")
            return

        # Step 2: Load the three-class mask dataset (if not in prediction-only mode)
        if not args.predict_only:
            print("\nStep 2: Loading three-class masks dataset...")
            X, Y = load_images_from_directory(
                IMAGE_FOLDER,
                THREE_CLASS_MASK_FOLDER,
                target_size=(args.image_size, args.image_size),
                mask_type="three_class",
                max_images=args.max_images
            )

            if len(X) == 0:
                print("No images loaded. Please check your image and mask paths.")
                return

            # Split data
            X_train, X_val, Y_train, Y_val = train_test_split(
                X, Y, test_size=0.2, random_state=42)

            print(f"Dataset loaded: {len(X)} images")
            print(f"Training: {len(X_train)}, Validation: {len(X_val)}")

            # Also load original instance masks for evaluation
            print("Loading instance masks for later evaluation...")
            _, instance_masks = load_images_from_directory(
                IMAGE_FOLDER,
                INSTANCE_MASK_FOLDER,
                target_size=(args.image_size, args.image_size),
                mask_type="instance",
                max_images=args.max_images
            )

            if len(instance_masks) > 0:
                _, _, _, instance_Y_val = train_test_split(
                    X, instance_masks, test_size=0.2, random_state=42)
            else:
                print("No instance masks found. Will not be able to evaluate instance segmentation.")
                instance_Y_val = None
        else:
            # In predict-only mode, we'll load validation data later
            X_train, Y_train = None, None
            X_val, Y_val = None, None
            instance_Y_val = None
            print("Skipping dataset loading for training (predict-only mode)")

        # Step 3: Build and train the three-class segmentation model
        if not args.skip_training and not args.predict_only:
            print("\nStep 3: Building and training three-class segmentation model...")

            # Build three-class U-Net with 3-channel input
            three_class_model = build_three_class_unet(
                input_shape=X_train[0].shape,
                filters_base=args.filters
            )

            if three_class_model is None:
                print("Failed to build model. Exiting...")
                return

            # Print model summary
            three_class_model.summary()

            # Train model
            history = train_three_class_model(
                three_class_model,
                X_train,
                Y_train,
                X_val,
                Y_val,
                epochs=args.epochs,
                batch_size=args.batch_size
            )

            if not history:
                print("Training failed. Exiting...")
                return

            # Plot training history
            plot_training_history(history)

        # Step 4: Generate three-class predictions and convert to instances
        print("\nStep 4: Loading best model and generating predictions...")

        # For predict-only mode, load validation data now
        if args.predict_only and X_val is None:
            print("Loading validation data for prediction...")
            X_val, _ = load_images_from_directory(
                IMAGE_FOLDER,
                MASK_FOLDER,  # Doesn't matter which masks, we just need images
                target_size=(args.image_size, args.image_size),
                mask_type="binary",
                max_images=args.max_images
            )

            if len(X_val) == 0:
                print("No validation images found. Exiting...")
                return

            # Also load instance masks for evaluation if available
            print("Loading instance masks for evaluation...")
            _, instance_Y_val = load_images_from_directory(
                IMAGE_FOLDER,
                INSTANCE_MASK_FOLDER,
                target_size=(args.image_size, args.image_size),
                mask_type="instance",
                max_images=args.max_images
            )

        # Load the best model safely
        best_model_path = os.path.join(MODELS_FOLDER, "best_three_class_model.keras")
        if os.path.exists(best_model_path):
            best_model = load_model_safely(best_model_path)
            if best_model is None:
                print("Failed to load best model. Exiting...")
                return
        else:
            print("No saved model found. You need to train first or provide a model.")
            return

        # Generate three-class predictions
        print(f"Generating predictions for {len(X_val)} validation images...")
        three_class_predictions = predict_three_class(best_model, X_val, batch_size=1)

        # Step 5: Extract instances from three-class predictions using improved watershed
        print("\nStep 5: Extracting instances from three-class predictions with improved watershed...")
        instance_predictions = process_predictions_to_instances(
            three_class_predictions,
            min_size=args.min_object_size
        )

        # Step 6: Evaluate and visualize results
        print("\nStep 6: Evaluating and visualizing results...")

        # Visualize predictions
        if args.visualize:
            visualize_three_class_predictions(
                X_val,
                three_class_predictions,
                instance_predictions,
                true_instances=instance_Y_val,
                max_samples_per_file=5
            )

        # Compute instance segmentation metrics if ground truth available
        if instance_Y_val is not None and len(instance_Y_val) > 0:
            metrics = compute_instance_metrics(instance_predictions, instance_Y_val)

            print("\nInstance Segmentation Metrics:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Total GT Instances: {metrics['gt_instances']}")
            print(f"  Total Predicted Instances: {metrics['pred_instances']}")

        # Save instance segmentation results
        np.save(os.path.join(RESULTS_FOLDER, "three_class_instance_predictions.npy"), instance_predictions)
        print(f"Instance segmentation results saved to {os.path.join(RESULTS_FOLDER, 'three_class_instance_predictions.npy')}")

        print("\nThree-class Instance Segmentation Complete!")
        print(f"Results saved in: {RESULTS_FOLDER}")

        return best_model

    except Exception as e:
        print(f"Error in main_three_class_approach: {e}")
        traceback.print_exc()
        return None

def create_dataset_directories():
    """
    Create all necessary dataset directories if they don't exist
    """
    directories = [
        IMAGE_FOLDER,
        MASK_FOLDER,
        CIRCULAR_MASK_FOLDER,
        INSTANCE_MASK_FOLDER,
        BOUNDARY_MASK_FOLDER,
        THREE_CLASS_MASK_FOLDER,
        RESULTS_FOLDER,
        MODELS_FOLDER
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Create dataset directories
    create_dataset_directories()

    # Run main pipeline
    main_three_class_approach()
