import shutil

import pandas as pd
import utils as ut
import os

import mediapipe as mp

METADATA_PATH: str = os.path.join(ut.get_dataset_path(), "metadata.csv")
ALL_IMAGES_PATH: str = os.path.join(ut.get_dataset_path(), "all_images")
OUTPUT_DIR: str = os.path.join(ut.get_dataset_path(), "PAD_UFES_FACES")

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

        if diagnostic in classes:
            label = classes[diagnostic]

            class_path = os.path.join(OUTPUT_DIR, label)
            os.makedirs(class_path, exist_ok=True)

            src = os.path.join(ALL_IMAGES_PATH, image_id)
            dst = os.path.join(class_path, image_id)

            if os.path.exists(src):
                shutil.copy(src, dst)
                count += 1

    print(f"All images placed in {OUTPUT_DIR}")


if __name__ == "__main__":
    place_images()

