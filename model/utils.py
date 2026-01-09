import os

is_colab: bool = False

def create_sample_directories():
    directories: list[str] = [
        '../dataset',
        '../check',
        '../exit',
        '../visualizations',
        '../model'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_model_path():
    model_path: str = "/content/model" if is_colab else '../model'
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    return model_path

def get_check_path():
    check_path: str = "/content/check" if is_colab else '../check'
    if not os.path.exists(check_path):
        os.makedirs(check_path, exist_ok=True)

    return check_path

def get_exit_path():
    exit_path: str = "/content/exit" if is_colab else '../exit'
    if not os.path.exists(exit_path):
        os.makedirs(exit_path, exist_ok=True)

    return exit_path

def get_visualizations_path():
    visualizations_path: str = "/content/visualizations" if is_colab else '../visualizations'
    if not os.path.exists(visualizations_path):
        os.makedirs(visualizations_path, exist_ok=True)

    return visualizations_path

def get_dataset_path():
    dataset_path: str = "/content/dataset" if is_colab else '../dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    return dataset_path

def get_dataset():
    dataset_path = os.path.join(get_dataset_path(), "PAD_UFES_FACES")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found in {dataset_path} path. Please download it first.")
    return dataset_path


def get_model_and_weights_name():
    return ("SkinAnalysis_AI.keras", "SkinAnalysis_AI_best_weights.keras")