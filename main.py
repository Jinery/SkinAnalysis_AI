import model.train as tr
import model.utils as ut
import model.dataset_sorter as ds


def main():
    # ut.is_colab = True # Uncomment if you using colab.
    ut.create_sample_directories()
    # ds.place_images() # uncomment for sort dataset(PAD_UFES_FACES)
    # ds.prepare_healthy_images(300) # uncomment for prepare healthy images for datase(UTKFace)
    tr.train_model()