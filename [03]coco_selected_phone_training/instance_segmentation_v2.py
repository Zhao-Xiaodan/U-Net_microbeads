
#!/usr/bin/env python3
"""
Complete implementation for bead instance segmentation
Uses a two-stage approach that is more reliable for this use case
Improved version with memory management and stability enhancements
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
from skimage import feature, morphology
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
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

# Ensure output folders exist
for folder in [RESULTS_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

#####################################
# PREPROCESSING FUNCTIONS           #
#####################################

def preprocess_image(img, disk_size=50):
    """
    Advanced preprocessing for microscopy images

    Args:
        img: Input grayscale image (0-1 float)
        disk_size: Size of structuring element for background extraction

    Returns:
        Preprocessed image (0-1 float with 2 channels)
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

        # Return multi-channel result
        return np.stack([enhanced / 255.0, log], axis=-1)
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        # Return original image with placeholder second channel
        return np.stack([img, np.zeros_like(img)], axis=-1)

def debug_instance_masks(mask_dir, num_samples=5):
    """
    Debug instance masks by loading and visualizing them

    Args:
        mask_dir: Directory containing instance masks
        num_samples: Number of samples to visualize
    """
    # Get all mask files
    mask_paths = glob(os.path.join(mask_dir, '*.png'))

    if len(mask_paths) == 0:
        print(f"ERROR: No instance masks found in {mask_dir}")
        return

    print(f"Found {len(mask_paths)} instance masks in {mask_dir}")

    # Show a few samples
    plt.figure(figsize=(15, 5 * min(num_samples, len(mask_paths))))

    for i, mask_path in enumerate(mask_paths[:num_samples]):
        try:
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if mask is None:
                print(f"Error loading mask: {mask_path}")
                continue

            # Print basic stats
            print(f"\nMask: {os.path.basename(mask_path)}")
            print(f"Shape: {mask.shape}")
            print(f"Data type: {mask.dtype}")
            print(f"Min value: {mask.min()}")
            print(f"Max value: {mask.max()}")
            print(f"Unique values: {np.unique(mask).tolist()}")

            # Convert to color visualization
            # Create a colormap for visualization
            mask_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            unique_instances = np.unique(mask)

            # Skip background (0)
            for idx, instance_id in enumerate(unique_instances[unique_instances > 0]):
                # Generate a color based on instance ID
                color = np.array([
                    (instance_id * 50) % 255,
                    (instance_id * 100) % 255,
                    (instance_id * 150) % 255
                ], dtype=np.uint8)

                # Apply the color to the mask
                mask_viz[mask == instance_id] = color

            # Display original grayscale
            plt.subplot(num_samples, 2, 2*i + 1)
            plt.imshow(mask, cmap='nipy_spectral')
            plt.title(f"Original Instance Mask")
            plt.colorbar()
            plt.axis('off')

            # Display colored visualization
            plt.subplot(num_samples, 2, 2*i + 2)
            plt.imshow(mask_viz)
            plt.title(f"Colored Instance Visualization")
            plt.axis('off')

        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "instance_mask_debug.png"))
    plt.close()

def convert_to_binary_masks(instance_masks):
    """
    Convert instance masks to binary masks (foreground vs background)

    Args:
        instance_masks: Array of instance masks

    Returns:
        Binary masks
    """
    binary_masks = []

    for mask in instance_masks:
        # Any non-zero value becomes 1
        binary = (mask > 0).astype(np.float32)
        binary_masks.append(binary)

    return np.array(binary_masks)

def load_images_from_directory(image_dir, mask_dir, target_size=(256, 256), mask_type="circular", max_images=None):
    """
    Load images and corresponding masks with optional type selection

    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing masks
        target_size: Size to resize images to
        mask_type: Type of mask to use - 'original', 'circular', 'instance', or 'boundary'
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
        if mask_type == "original":
            mask_dir = MASK_FOLDER
        elif mask_type == "circular":
            mask_dir = CIRCULAR_MASK_FOLDER
        elif mask_type == "boundary":
            mask_dir = BOUNDARY_MASK_FOLDER
        elif mask_type == "instance":
            mask_dir = INSTANCE_MASK_FOLDER

        print(f"Using masks from: {mask_dir}")

        for image_path in image_paths:
            try:
                # Load image
                img = load_img(image_path, color_mode="grayscale", target_size=target_size)
                img = img_to_array(img) / 255.0

                # Apply preprocessing
                processed_img = preprocess_image(img.squeeze())

                # Get corresponding mask filename
                image_name = os.path.basename(image_path)
                mask_path = os.path.join(mask_dir, image_name)

                # For instance masks, use PNG extension
                if mask_type == "instance":
                    mask_path = os.path.splitext(mask_path)[0] + ".png"

                if os.path.exists(mask_path):
                    # Load mask
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        print(f"Warning: Failed to load mask: {mask_path}")
                        continue

                    # Resize to target size
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

                    # Normalize
                    if mask_type == "instance":
                        # For instance masks, preserve instance IDs
                        # Just resize, but keep original values
                        mask = mask.astype(np.float32)
                    else:
                        # For binary masks, convert to 0-1
                        mask = (mask > 127).astype(np.float32)

                    # Add channel dimension
                    mask = np.expand_dims(mask, axis=-1)

                    images.append(processed_img)
                    masks.append(mask)

                    print(f"Processed {image_name}")
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

# Define losses and metrics
def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss for overlap quality
    """
    # Ensure consistent shapes
    y_true = tf.cast(y_true, y_pred.dtype)

    # Flatten tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

def f1_score(y_true, y_pred, threshold=0.5):
    """F1 score metric"""
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)

    # Calculate precision and recall
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1

def iou_score(y_true, y_pred, threshold=0.5):
    """IoU score metric"""
    y_true = tf.cast(y_true > threshold, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    # Calculate IoU
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou

# Build an enhanced U-Net for binary segmentation
def build_enhanced_unet(input_shape=(256, 256, 2), filters_base=32):
    """
    Build an enhanced U-Net model for the first stage (binary segmentation)

    Args:
        input_shape: Input tensor shape (H, W, C)
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

        # Bridge
        conv4 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(conv4)

        # Decoder (upsampling path)
        # Level 3
        up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = layers.Conv2D(filters_base*4, 2, activation='relu', padding='same')(up5)
        merge5 = layers.concatenate([conv3, up5], axis=3)
        conv5 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(merge5)
        conv5 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(conv5)

        # Level 2
        up6 = layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = layers.Conv2D(filters_base*2, 2, activation='relu', padding='same')(up6)
        merge6 = layers.concatenate([conv2, up6], axis=3)
        conv6 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(conv6)

        # Level 1
        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = layers.Conv2D(filters_base, 2, activation='relu', padding='same')(up7)
        merge7 = layers.concatenate([conv1, up7], axis=3)
        conv7 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(conv7)

        # Output layer
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

        # Create and compile model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=dice_loss,
            metrics=['accuracy', f1_score, iou_score]
        )

        return model
    except Exception as e:
        print(f"Error in build_enhanced_unet: {e}")
        return None

#####################################
# IMPROVED TRAINING LOOP           #
#####################################

def train_with_careful_loop(model, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=2):
    """
    Train the model using a careful training loop with improved stability

    Args:
        model: The compiled model to train
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        epochs: Number of epochs to train
        batch_size: Batch size for training

    Returns:
        history: Dictionary containing training history
    """
    # Set up optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = dice_loss

    # Training history tracking
    history = {
        'loss': [],
        'val_loss': [],
        'val_f1_score': [],
        'val_iou_score': []
    }

    # Best model tracking
    best_val_f1 = 0
    best_epoch = 0

    try:
        print("Starting careful training loop...")

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
                    print("  Forward pass...")
                    predictions = model(X_batch, training=True)
                    print("  Computing loss...")
                    loss = loss_fn(Y_batch, predictions)
                    print(f"  Loss: {loss.numpy():.4f}")

                print("  Computing gradients...")
                gradients = tape.gradient(loss, model.trainable_variables)
                print("  Applying gradients...")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                epoch_loss += loss.numpy()
                batch_count += 1

                # Force memory cleanup after each batch
                gc.collect()

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count
            history['loss'].append(avg_epoch_loss)

            # Manual validation with single sample prediction for stability
            print("Evaluating on validation data...")
            val_losses = []
            val_f1_scores = []
            val_iou_scores = []

            for i in range(0, len(X_val), batch_size):
                end = min(i + batch_size, len(X_val))
                X_batch_val = X_val[i:end]
                Y_batch_val = Y_val[i:end]

                print(f"  Processing validation batch {i//batch_size + 1}/{(len(X_val)-1)//batch_size + 1}")

                # Process each validation sample individually for more stability
                for j in range(len(X_batch_val)):
                    X_sample = np.expand_dims(X_batch_val[j], axis=0)
                    Y_sample = np.expand_dims(Y_batch_val[j], axis=0)

                    # Prediction in non-training mode
                    pred = model(X_sample, training=False)

                    # Calculate validation metrics
                    val_loss = loss_fn(Y_sample, pred).numpy()
                    val_losses.append(val_loss)

                    # Calculate F1 score
                    sample_f1 = f1_score(Y_sample, pred).numpy()
                    val_f1_scores.append(sample_f1)

                    # Calculate IoU score
                    sample_iou = iou_score(Y_sample, pred).numpy()
                    val_iou_scores.append(sample_iou)

                # Clean up memory
                gc.collect()

            # Average validation metrics
            avg_val_loss = np.mean(val_losses)
            avg_val_f1 = np.mean(val_f1_scores)
            avg_val_iou = np.mean(val_iou_scores)

            # Update history
            history['val_loss'].append(avg_val_loss)
            history['val_f1_score'].append(avg_val_f1)
            history['val_iou_score'].append(avg_val_iou)

            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f} - val_loss: {avg_val_loss:.4f} - val_f1: {avg_val_f1:.4f} - val_iou: {avg_val_iou:.4f}")

            # Save best model
            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                best_epoch = epoch

                # Save best model
                try:
                    print("Saving best model...")
                    model.save(os.path.join(MODELS_FOLDER, "best_binary_model.keras"))
                    print(f"Model saved to {os.path.join(MODELS_FOLDER, 'best_binary_model.keras')}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                    traceback.print_exc()

            # Memory cleanup
            gc.collect()

        print(f"Training completed. Best model from epoch {best_epoch+1} with F1 score: {best_val_f1:.4f}")
        return history

    except Exception as e:
        print(f"Error in training loop: {e}")
        traceback.print_exc()
        return history

#####################################
# STABLE PREDICTION FUNCTIONS       #
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
                'dice_loss': dice_loss,
                'f1_score': f1_score,
                'iou_score': iou_score
            },
            compile=False
        )

        # Compile after loading
        print("Compiling model...")
        model.compile(
            optimizer='adam',  # Use a simpler optimizer
            loss=dice_loss,
            metrics=['accuracy', f1_score, iou_score]
        )

        print("Successfully loaded and compiled model")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def predict_safely(model, images, batch_size=1):
    """
    Generate predictions with enhanced safety measures to prevent segmentation faults

    Args:
        model: Loaded model
        images: Images to predict
        batch_size: Batch size for prediction (use 1 for maximum stability)

    Returns:
        Predictions
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
                empty_pred = np.zeros(img.shape[:-1] + (1,))
                predictions.append(empty_pred)
                continue

        return np.array(predictions)

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        traceback.print_exc()
        return np.array(predictions)

#####################################
# INSTANCE SEGMENTATION FUNCTIONS   #
#####################################

def extract_instances_from_binary(binary_mask, min_size=30, min_distance=10):
    """
    Extract individual instances from a binary mask using watershed

    Args:
        binary_mask: Binary mask (foreground/background)
        min_size: Minimum size of objects to keep
        min_distance: Minimum distance between markers

    Returns:
        Instance mask with unique IDs for each object
    """
    try:
        # Ensure binary mask is boolean
        binary = binary_mask.astype(bool)

        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=min_size)

        # Distance transform
        distance = ndi.distance_transform_edt(binary)

        # Find local maxima (markers for watershed)
        coords = feature.peak_local_max(distance, min_distance=min_distance, labels=binary)

        # Create markers for watershed
        markers = np.zeros_like(binary, dtype=np.int32)
        if len(coords) > 0:
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

            # Apply watershed
            labeled_mask = watershed(-distance, markers, mask=binary)
        else:
            # No peaks found, just use connected components
            labeled_mask = morphology.label(binary)

        return labeled_mask
    except Exception as e:
        print(f"Error in extract_instances_from_binary: {e}")
        traceback.print_exc()
        return morphology.label(binary)

def convert_binary_predictions_to_instances(binary_predictions, min_size=30, min_distance=10):
    """
    Convert a batch of binary predictions to instance masks

    Args:
        binary_predictions: Binary predictions from segmentation model
        min_size: Minimum size of objects to keep
        min_distance: Minimum distance between markers

    Returns:
        Instance masks with unique IDs for each object
    """
    instance_masks = []

    for i, pred in enumerate(binary_predictions):
        try:
            print(f"Processing prediction {i+1}/{len(binary_predictions)} to extract instances...")

            # Threshold to get binary mask
            binary = (pred.squeeze() > 0.5).astype(np.uint8)

            # Extract instances
            instances = extract_instances_from_binary(binary, min_size=min_size, min_distance=min_distance)

            # Add to list
            instance_masks.append(instances)

            # Force memory cleanup periodically
            if (i+1) % 10 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error converting prediction {i+1} to instances: {e}")
            # Add an empty mask as placeholder
            instance_masks.append(np.zeros_like(binary, dtype=np.int32))

    return np.array(instance_masks)

def generate_boundary_masks(instance_masks, kernel_size=3):
    """Generate boundary masks from instance masks"""
    boundary_masks = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for mask in instance_masks:
        # Dilate each instance separately and find boundaries
        unique_instances = np.unique(mask)[1:]  # Skip background
        boundary_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for instance_id in unique_instances:
            instance = (mask == instance_id).astype(np.uint8)
            dilated = cv2.dilate(instance, kernel)
            boundary = dilated - instance
            boundary_mask = np.logical_or(boundary_mask, boundary)
        
        boundary_masks.append(boundary_mask.astype(np.float32))
    
    return np.array(boundary_masks)

def train_model(model, train_images, train_masks, val_images, val_masks, epochs=50, batch_size=8):
    """Train the model with both binary segmentation and boundary detection"""
    # Generate boundary masks
    train_boundaries = generate_boundary_masks(train_masks)
    val_boundaries = generate_boundary_masks(val_masks)
    
    # Convert instance masks to binary masks for training
    train_binary = convert_to_binary_masks(train_masks)
    val_binary = convert_to_binary_masks(val_masks)
    
    # Train the model
    history = model.fit(
        train_images,
        {'binary_output': train_binary, 'boundary_output': train_boundaries},
        validation_data=(
            val_images,
            {'binary_output': val_binary, 'boundary_output': val_boundaries}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(MODELS_FOLDER, 'best_multitask_model.h5'),
                monitor='val_binary_output_f1_score',
                mode='max',
                save_best_only=True
            )
        ]
    )
    
    return history

def evaluate_instance_segmentation(binary_preds, true_instances, prefix="instance_eval", min_size=30, min_distance=10):
    """
    Evaluate instance segmentation quality and visualize results

    Args:
        binary_preds: Binary predictions from the model
        true_instances: Ground truth instance masks
        prefix: Prefix for output files
        min_size: Minimum size of objects to keep
        min_distance: Minimum distance between markers
    """
    # Convert binary predictions to instances
    print("Converting binary predictions to instance masks...")
    pred_instances = convert_binary_predictions_to_instances(binary_preds, min_size=min_size, min_distance=min_distance)

    # Visualization
    num_samples = min(5, len(binary_preds))

    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        try:
            # Original binary prediction
            plt.subplot(num_samples, 3, 3*i + 1)
            plt.imshow(binary_preds[i].squeeze(), cmap='gray')
            plt.title(f"Binary Pred {i+1}")
            plt.axis('off')

            # Ground truth instances
            plt.subplot(num_samples, 3, 3*i + 2)
            plt.imshow(true_instances[i].squeeze(), cmap='nipy_spectral')
            plt.title(f"GT Instances {i+1}")
            plt.axis('off')

            # Predicted instances
            plt.subplot(num_samples, 3, 3*i + 3)
            plt.imshow(pred_instances[i], cmap='nipy_spectral')
            plt.title(f"Pred Instances {i+1}")
            plt.axis('off')
        except Exception as e:
            print(f"Error visualizing sample {i+1}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{prefix}_instance_comparison.png"))
    plt.close()

    # Compute basic metrics
    metrics = {}
    tp_total, fp_total, fn_total = 0, 0, 0
    gt_count_total, pred_count_total = 0, 0

    # Count correct instances (very simplified)
    for i in range(len(true_instances)):
        try:
            gt_count = len(np.unique(true_instances[i])) - 1  # Subtract background
            pred_count = len(np.unique(pred_instances[i])) - 1  # Subtract background
            gt_count_total += gt_count
            pred_count_total += pred_count

            # Very simple overlap metric (can be improved with IoU-based matching)
            # This is just a counting-based approximation
            tp = min(gt_count, pred_count)  # True positives (approximate)
            fp = max(0, pred_count - gt_count)  # False positives
            fn = max(0, gt_count - pred_count)  # False negatives

            tp_total += tp
            fp_total += fp
            fn_total += fn

            print(f"Sample {i+1}: GT instances: {gt_count}, Predicted instances: {pred_count}")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
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

    print(f"\nAggregate Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Total GT Instances: {gt_count_total}")
    print(f"  Total Predicted Instances: {pred_count_total}")

    # Save instance segmentation results
    np.save(os.path.join(RESULTS_FOLDER, f"{prefix}_instance_predictions.npy"), pred_instances)
    print(f"Instance segmentation results saved to {os.path.join(RESULTS_FOLDER, f'{prefix}_instance_predictions.npy')}")

    return metrics, pred_instances

#####################################
# VISUALIZATION FUNCTIONS           #
#####################################

def visualize_predictions(images, binary_preds, instance_preds, max_samples=5, prefix="visualization"):
    """
    Create comprehensive visualizations of the segmentation results

    Args:
        images: Original preprocessed images
        binary_preds: Binary predictions from the model
        instance_preds: Instance mask predictions
        max_samples: Maximum number of samples to visualize
        prefix: Prefix for output files
    """
    num_samples = min(max_samples, len(images))

    plt.figure(figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        try:
            # Original image (first channel of preprocessed image)
            plt.subplot(num_samples, 4, 4*i + 1)
            plt.imshow(images[i][:,:,0], cmap='gray')
            plt.title(f"Original Image {i+1}")
            plt.axis('off')

            # Second channel (LoG)
            plt.subplot(num_samples, 4, 4*i + 2)
            plt.imshow(images[i][:,:,1], cmap='gray')
            plt.title(f"LoG Filter {i+1}")
            plt.axis('off')

            # Binary prediction
            plt.subplot(num_samples, 4, 4*i + 3)
            plt.imshow(binary_preds[i].squeeze(), cmap='gray')
            plt.title(f"Binary Prediction {i+1}")
            plt.axis('off')

            # Instance prediction
            plt.subplot(num_samples, 4, 4*i + 4)
            plt.imshow(instance_preds[i], cmap='nipy_spectral')
            plt.title(f"Instance Prediction {i+1}\n({len(np.unique(instance_preds[i]))-1} objects)")
            plt.axis('off')

        except Exception as e:
            print(f"Error visualizing sample {i+1}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{prefix}_full_visualization.png"))
    plt.close()

def export_instance_overlays(images, instance_preds, max_samples=None, prefix="results"):
    """
    Export instance segmentation overlays on original images

    Args:
        images: Original preprocessed images
        instance_preds: Instance mask predictions
        max_samples: Maximum number of samples to export (None for all)
        prefix: Prefix for output files
    """
    if max_samples is None:
        max_samples = len(images)

    num_samples = min(max_samples, len(images))

    # Create output directory
    overlay_dir = os.path.join(RESULTS_FOLDER, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"Exporting {num_samples} instance segmentation overlays...")

    for i in range(num_samples):
        try:
            # Get original image (first channel)
            orig_img = images[i][:,:,0]

            # Convert to RGB
            rgb_img = np.stack([orig_img, orig_img, orig_img], axis=-1)

            # Convert to uint8 for OpenCV
            rgb_img = (rgb_img * 255).astype(np.uint8)

            # Create instance overlay
            overlay = rgb_img.copy()

            # Get unique instance IDs
            unique_instances = np.unique(instance_preds[i])

            # Skip background (0)
            for instance_id in unique_instances[unique_instances > 0]:
                # Generate a color based on instance ID
                color = np.array([
                    (instance_id * 50) % 255,
                    (instance_id * 100) % 255,
                    (instance_id * 150) % 255
                ], dtype=np.uint8)

                # Create mask for this instance
                instance_mask = (instance_preds[i] == instance_id)

                # Apply color with alpha blending
                overlay[instance_mask] = (
                    overlay[instance_mask] * 0.5 +
                    color * 0.5
                ).astype(np.uint8)

                # Draw contour around the instance
                contours, _ = cv2.findContours(
                    instance_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

            # Save overlay
            cv2.imwrite(
                os.path.join(overlay_dir, f"{prefix}_overlay_{i+1}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

            print(f"Saved overlay for sample {i+1}")

        except Exception as e:
            print(f"Error exporting overlay for sample {i+1}: {e}")

    print(f"Overlays saved to {overlay_dir}")

def calculate_advanced_metrics(pred_instances, true_instances):
    """
    Calculate advanced instance segmentation metrics

    Args:
        pred_instances: Predicted instance masks
        true_instances: Ground truth instance masks

    Returns:
        Dictionary of metrics
    """
    # This is a placeholder for more advanced metrics calculation
    # For a real implementation, you would implement:
    # 1. Instance-level IoU calculation
    # 2. Precision, recall at different IoU thresholds
    # 3. Average Precision (AP) calculation
    # 4. Panoptic Quality metrics

    metrics = {
        'note': 'Basic count-based metrics only. For production use, implement proper instance matching with IoU.'
    }

    return metrics

#####################################
# MAIN FUNCTION                     #
#####################################

def main_two_stage_approach():
    """
    Main function using a two-stage approach for instance segmentation
    1. Train a binary segmentation model
    2. Convert binary masks to instance masks using watershed
    """
    try:
        print("=" * 50)
        print("TWO-STAGE INSTANCE SEGMENTATION")
        print("=" * 50)

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train a two-stage instance segmentation model')
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
        parser.add_argument('--min-distance', type=int, default=10,
                          help='Minimum distance between objects for watershed')
        parser.add_argument('--export-overlays', action='store_true',
                          help='Export instance segmentation overlays on original images')
        parser.add_argument('--visualize', action='store_true', default=True,
                          help='Create visualizations of the results')
        args = parser.parse_args()

        # Step 1: Debug instance masks first (if not in prediction-only mode)
        if not args.predict_only:
            print("\nStep 1: Debugging instance masks...")
            debug_instance_masks(INSTANCE_MASK_FOLDER)

        # Step 2: Load instance masks (if not in prediction-only mode)
        if not args.predict_only:
            print("\nStep 2: Loading instance masks...")
            X, instance_Y = load_images_from_directory(
                IMAGE_FOLDER,
                INSTANCE_MASK_FOLDER,
                target_size=(args.image_size, args.image_size),
                mask_type="instance",
                max_images=args.max_images
            )

            if len(X) == 0:
                print("No images loaded. Please check your image and mask paths.")
                return

            print("Converting instance masks to binary masks for first-stage training...")
            binary_Y = convert_to_binary_masks(instance_Y)

            # Split data
            X_train, X_val, binary_Y_train, binary_Y_val = train_test_split(
                X, binary_Y, test_size=0.2, random_state=42)

            # Also split the instance masks for later evaluation
            _, _, instance_Y_train, instance_Y_val = train_test_split(
                X, instance_Y, test_size=0.2, random_state=42)

            print(f"Dataset loaded: {len(X)} images")
            print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
        else:
            # In predict-only mode, we'll load validation data later
            X_train, binary_Y_train = None, None
            X_val, binary_Y_val = None, None
            instance_Y_val = None
            print("Skipping dataset loading for training (predict-only mode)")

        # Step 3: Build and train binary segmentation model (if not skipping training)
        if not args.skip_training and not args.predict_only:
            print("\nStep 3: Building and training binary segmentation model...")

            # Build enhanced U-Net for binary segmentation
            binary_model = build_enhanced_unet(
                input_shape=X_train[0].shape,
                filters_base=args.filters
            )

            if binary_model is None:
                print("Failed to build model. Exiting...")
                return

            # Print model summary
            binary_model.summary()

            # Train binary segmentation model
            history = train_with_careful_loop(
                binary_model,
                X_train,
                binary_Y_train,
                X_val,
                binary_Y_val,
                epochs=args.epochs,
                batch_size=args.batch_size
            )

            if not history:
                print("Training failed. Exiting...")
                return

            # Plot training history
            print("\nPlotting training history...")

            plt.figure(figsize=(15, 5))

            # Loss plot
            plt.subplot(1, 3, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # F1 score plot
            plt.subplot(1, 3, 2)
            plt.plot(history['val_f1_score'], label='Validation F1')
            plt.title('F1 Score History')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()

            # IoU plot
            plt.subplot(1, 3, 3)
            plt.plot(history['val_iou_score'], label='Validation IoU')
            plt.title('IoU History')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_FOLDER, "binary_training_history.png"))
            plt.close()

        # Step 4: Generate binary predictions and convert to instances
        print("\nStep 4: Loading best model and generating predictions...")

        # For predict-only mode, load validation data now
        if args.predict_only and instance_Y_val is None:
            print("Loading validation data for prediction...")
            _, instance_Y = load_images_from_directory(
                IMAGE_FOLDER,
                INSTANCE_MASK_FOLDER,
                target_size=(args.image_size, args.image_size),
                mask_type="instance",
                max_images=args.max_images
            )

            if len(instance_Y) > 0:
                # Create a small validation set
                X_val, instance_Y_val = train_test_split(
                    np.zeros_like(instance_Y), instance_Y, test_size=0.2, random_state=42)

                # Just keep instance_Y_val
                X_val = None
                binary_Y_val = convert_to_binary_masks(instance_Y_val)
                print(f"Loaded {len(instance_Y_val)} validation instances")
            else:
                print("No validation instances found. Exiting...")
                return

        # Load the best model safely
        best_model_path = os.path.join(MODELS_FOLDER, "best_binary_model.keras")
        if os.path.exists(best_model_path):
            best_model = load_model_safely(best_model_path)
            if best_model is None:
                print("Failed to load best model. Exiting...")
                return
        else:
            print("No saved model found. You need to train first or provide a model.")
            return

        # Load validation images if we don't have them yet
        if X_val is None:
            print("Loading validation images for prediction...")
            X_val, _ = load_images_from_directory(
                IMAGE_FOLDER,
                MASK_FOLDER,  # Doesn't matter, we just need the images
                target_size=(args.image_size, args.image_size),
                mask_type="original",
                max_images=args.max_images
            )

            if len(X_val) == 0:
                print("No validation images found. Exiting...")
                return

            # Create a smaller validation set if needed
            if len(X_val) > 20:
                print(f"Using a subset of {min(20, len(X_val))} images for prediction...")
                X_val = X_val[:min(20, len(X_val))]

        # Generate binary predictions for validation set
        print(f"Generating predictions for {len(X_val)} validation images...")
        binary_predictions = predict_safely(best_model, X_val, batch_size=1)

        # Step 5: Evaluate instance segmentation performance
        print("\nStep 5: Evaluating instance segmentation performance...")

        if instance_Y_val is not None and len(instance_Y_val) > 0:
            # Make sure we have the right number of ground truth instances
            if len(instance_Y_val) > len(binary_predictions):
                instance_Y_val = instance_Y_val[:len(binary_predictions)]

            metrics, pred_instances = evaluate_instance_segmentation(
                binary_predictions,
                instance_Y_val,
                prefix="two_stage",
                min_size=args.min_object_size,
                min_distance=args.min_distance
            )
        else:
            print("No ground truth instance masks available for evaluation.")
            print("Converting binary predictions to instances anyway...")
            pred_instances = convert_binary_predictions_to_instances(
                binary_predictions,
                min_size=args.min_object_size,
                min_distance=args.min_distance
            )

            # Save instance segmentation results
            np.save(os.path.join(RESULTS_FOLDER, "instance_predictions.npy"), pred_instances)
            print(f"Instance segmentation results saved to {os.path.join(RESULTS_FOLDER, 'instance_predictions.npy')}")

        # Step 6: Post-process and refine instance segmentation
        print("\nStep 6: Refining instance segmentation...")

        # Apply custom minimum size parameter if specified
        if args.min_object_size > 0:
            print(f"Removing objects smaller than {args.min_object_size} pixels...")
            for i in range(len(pred_instances)):
                try:
                    # Get unique instance IDs
                    unique_instances = np.unique(pred_instances[i])

                    # Skip background (0)
                    for instance_id in unique_instances[unique_instances > 0]:
                        # Get size of this instance
                        instance_size = np.sum(pred_instances[i] == instance_id)

                        # If smaller than threshold, remove
                        if instance_size < args.min_object_size:
                            pred_instances[i][pred_instances[i] == instance_id] = 0
                            print(f"  Removed small object (ID: {instance_id}, Size: {instance_size})")
                except Exception as e:
                    print(f"Error refining instance mask {i}: {e}")

        # Step 7: Visualize and export results
        if args.visualize:
            print("\nStep 7: Creating visualizations...")
            visualize_predictions(
                X_val,
                binary_predictions,
                pred_instances,
                max_samples=10,
                prefix="final_results"
            )

        # Export instance overlays if requested
        if args.export_overlays:
            print("\nStep 8: Exporting instance segmentation overlays...")
            export_instance_overlays(
                X_val,
                pred_instances,
                max_samples=None,  # Export all
                prefix="instance_segmentation"
            )

        print("Instance Segmentation Complete!")
        print(f"Results saved in: {RESULTS_FOLDER}")

        return best_model
    except Exception as e:
        print(f"Error in main_two_stage_approach: {e}")
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
    main_two_stage_approach()
