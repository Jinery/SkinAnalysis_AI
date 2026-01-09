import shutil

import pandas as pd
import utils as ut
import os

import mediapipe as mp
import cv2

METADATA_PATH: str = os.path.join(ut.get_dataset_path(), "metadata.csv")
ALL_IMAGES_PATH: str = os.path.join(ut.get_dataset_path(), "all_images")
OUTPUT_DIR: str = os.path.join(ut.get_dataset_path(), "PAD_UFES_FACES")

SOURCE_FACES_DIR: str = os.path.join(ut.get_dataset_path(), "UTKFace")
TARGET_FACES_DIR: str = os.path.join(OUTPUT_DIR, "healthy")

classes: dict[str, str] = {
    "BCC": "problem",
    "ACK": "problem",
    "NEV": "nevus",
    "SEK": "problem",
    "SCC": "problem",
}

model_path = "blaze_face_short_range.tflite"
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

def place_images():

    if not os.path.exists(METADATA_PATH):
        print("Metadata not found.")
        exit(-1)

    if not os.path.exists(ALL_IMAGES_PATH):
        print("All images folder not found.")
        exit(-1)

    df = pd.read_csv(METADATA_PATH)
    count: int = 0

    for index, row in df.iterrows():
        image_id = row['img_id']
        diagnostic = row['diagnostic']
        region = row['region']

        if diagnostic in classes and isinstance(region, str):
            if region.upper() == "FACE":
                label = classes[diagnostic]

                class_path = os.path.join(OUTPUT_DIR, label)
                os.makedirs(class_path, exist_ok=True)

                src = os.path.join(ALL_IMAGES_PATH, image_id)
                dst = os.path.join(class_path, image_id)

                if os.path.exists(src):
                    shutil.copy(src, dst)
                    count += 1

    print(f"All images placed in {OUTPUT_DIR}")

def save_patch(img, x, y, size, name):
    patch = img[y:y+size, x:x+size]
    if patch.shape[0] == size and patch.shape[1] == size:
        cv2.imwrite(os.path.join(TARGET_FACES_DIR, name), patch)
        return True
    return False

def prepare_healthy_images(limit: int):

    if not os.path.exists(TARGET_FACES_DIR):
        os.makedirs(TARGET_FACES_DIR)

    face_count: int = 0
    patch_count: int = 0
    files = os.listdir(SOURCE_FACES_DIR)
    with FaceDetector.create_from_options(options) as detector:
        for file in files:
            if face_count >= limit: break

            img_path = os.path.join(SOURCE_FACES_DIR, file)
            image = cv2.imread(img_path)
            if image is None: continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            detection_result = detector.detect(mp_image)
            if detection_result.detections:
                bbox = detection_result.detections[0].bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                if save_patch(image, x + w // 4, y + h // 10, w // 2, f"h_{patch_count}.png"):
                    patch_count += 1

                if save_patch(image, x + w // 10, y + h // 2, w // 3, f"h_{patch_count}.png"):
                    patch_count += 1

                if save_patch(image, x + w - w // 3 - w // 10, y + h // 2, w // 3, f"h_{patch_count}.png"):
                    patch_count += 1

                face_count += 1

        print(f"{face_count} images croped and placed in {OUTPUT_DIR}")


if __name__ == "__main__":
    place_images()
    prepare_healthy_images(300)

