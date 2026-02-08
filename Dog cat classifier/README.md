# Dog vs Cat Classifier

Simple convolutional neural network (CNN) to classify cat and dog images using TensorFlow/Keras. Training is done in the notebook `basic.ipynb` with data loaded from a local folder structure.

## Project Structure

- `basic.ipynb`: Training notebook (data loading, augmentation, model definition, training).
- `test.ipynb`: Testing/inference notebook.
- `data/`: Image dataset directory.
- `best_from_scratch.keras`: Best checkpoint saved during training.
- `cats_vs_dogs_basic_model.h5`: Saved model (H5 format).
- `cats_vs_dogs_basic_model.keras`: Saved model (Keras format).

## Dataset Layout

The notebook expects this directory structure:

```text
data/
  training_set/
    training_set/
      cats/
      dogs/
  test_set/
    test_set/
      cats/
      dogs/
```

## Training (Notebook)

Open `basic.ipynb` and run all cells. The pipeline:

- Loads images with `image_dataset_from_directory` at 224x224
- Scales pixels to [0, 1]
- Applies basic data augmentation
- Trains a small CNN and saves the best model to `best_from_scratch.keras`

## Notes

- Batch size is 32 and image size is 224x224 in the current notebook.
- Training uses `sparse_categorical_crossentropy` and `softmax` with 2 classes.

## Requirements

- Python 3.x
- TensorFlow
- Matplotlib

Install with:

```bash
pip install tensorflow matplotlib
```
