from io import BytesIO
import os
import requests
from PIL import Image
import utils as ut
import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_NAME, WEIGHTS_NAME = ut.get_model_and_weights_name() # Get model and weights names from utilities module.
MODEL_PATH: str = os.path.join(ut.get_exit_path(), MODEL_NAME) # Construct full path to the saved model.

# Image dimensions expected by the model (must match training size).
IMG_SIZE = (224, 224)
# Class labels for predictions (order must match training).
CLASS_NAMES = ["healthy", "nevus", "problem"]


def load_model() -> tf.keras.Model:
    """
    Load the pre-trained Keras model from disk.

    Returns:
        tf.keras.Model: Loaded model ready for inference

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For any other loading errors
    """
    try:
        # Check if model file exists before attempting to load.
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH} path.")

        # Load the complete model.
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model successfully loaded.")
        return model
    except Exception as e:
        print(f"Error on loading model: {e}")
        raise


def preprocess_image(image_path: str):
    """
    Preprocess an image for model prediction.
    Handles both local files and URLs.

    Args:
        image_path (str): Path to image or URL

    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 224, 224, 3)

    Raises:
        Exception: If preprocessing fails
    """
    try:
        # Handle URL inputs - download image from web.
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))  # Create PIL Image from bytes.
        else:
            # Handle local file inputs.
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            img = Image.open(image_path)

        # Ensure image is in RGB format.
        if img.mode != 'RGB':
            img = img.convert('RGB')


        img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)  # Resize to match model's expected input dimensions.
        img_array = np.asarray(img) # Convert to numpy array and normalize to 0-255 range.
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension.

        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")


def try_predict(model, processed_image):
    """
    Run model prediction on preprocessed image.

    Args:
        model (tf.keras.Model): Loaded model
        processed_image (numpy.ndarray): Preprocessed image array

    Returns:
        tuple: (predicted_class, confidence, top_predictions)
            - predicted_class (str): Name of top predicted class
            - confidence (float): Confidence score as percentage
            - top_predictions (list): Top 3 predictions with confidences
    """
    # Get predictions from model (verbose=0 suppresses progress bar).
    predictions = model.predict(processed_image, verbose=0)
    pred_row = predictions[0]  # Extract predictions for single image.


    predicted_index = int(np.argmax(pred_row)) # Get index of highest probability.
    predicted_class = CLASS_NAMES[predicted_index] # Map index to class name.
    confidence = float(pred_row[predicted_index]) * 100 # Convert probability to percentage.

    # Get indices of top 3 predictions (highest to lowest).
    top_indices = np.argsort(pred_row)[-3:][::-1]

    # Create list of top predictions with names and confidences.
    top_predictions = [(str(CLASS_NAMES[i]), float(pred_row[i]) * 100) for i in top_indices]

    return predicted_class, confidence, top_predictions


def display_prediction(image_path, predicted_class, confidence, all_predictions):
    """
    Display prediction results in formatted console output.

    Args:
        image_path (str): Path to original image
        predicted_class (str): Predicted class name
        confidence (float): Confidence percentage
        all_predictions (list): All top predictions
    """
    print(f"Image: {os.path.basename(image_path)}")
    print("=" * 25)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 25)
    print("Other predictions:")
    for index, (class_name, conf) in enumerate(all_predictions):
        print(f"* {index} {class_name}: {conf:.2f}%")


def test_model_with_images(image_paths):
    """
    Test model on multiple images and display results.

    Args:
        image_paths (list): List of image paths to test
    """
    print("Testing Model")
    print("=" * 40)

    try:
        model = load_model() # Load model once for all predictions.
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Process each image in the list.
    for img_path in image_paths:
        try:
            # Preprocess and predict.
            processed_image = preprocess_image(img_path)
            predicted_class, confidence, top_predictions = try_predict(
                model, processed_image
            )

            display_prediction(img_path, predicted_class, confidence, top_predictions) # Display formatted results.

        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("\nTesting completed")


def predict_single_image(image_path):
    """
    Predict a single image and return structured results.
    Intended for API or programmatic use.

    Args:
        image_path (str): Path to image file or URL

    Returns:
        dict: Prediction results or error information
    """
    try:
        model = load_model()
        processed_image = preprocess_image(image_path)
        predicted_class, confidence, top_predictions = try_predict(
            model, processed_image
        )

        # Return structured dictionary for easy JSON serialization.
        return {
            'success': True,
            'image': os.path.basename(image_path),
            'prediction': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    except Exception as e:
        # Return error information if prediction fails.
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    check_pictures_path = Path(ut.get_check_path()) # Get path to directory containing test images.

    # Collect all image files from the directory.
    test_images = [
        str(p) for p in check_pictures_path.iterdir()
        if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp")
    ]

    test_model_with_images(test_images) # Run model testing on all collected images.