# Image Processing and Augmentation Pipeline for CNN based Emotional Recognition

This project currently consists of a collection of scripts designed to process, augment, and analyze image datasets, particularly focusing on datasets categorized based on emotions. It leverages popular Python libraries for image manipulation and analysis, ensuring that the images are prepared for machine learning tasks, particularly in emotion recognition.

## Scripts Overview

- **renames_images.py**: Renames and organizes images into categories based on the presence of detected faces and their associated emotions.

- **normalize.py**: Normalizes image sizes and color formats, ensuring consistent dimensions and color representations across the dataset.

- **augment.py**: Enhances the dataset by applying various image transformations such as flipping and rotating.

- **generate_plots.py**: Provides visual insights into the dataset through class distribution charts, sample images, and pixel intensity histograms.

- **shuffle_plots.py**: Similar to `generate_plots.py`, but focuses on shuffling images and analyzing RGB pixel distributions to better understand the dataset's diversity.

- **generate_sets.py**: Generates training, validation, and testing sets for CNN training using preprocessed and augmented data.
  
- **training.py**: Trains convolutional neural network models for emotion recognition using preprocessed datasets. Models are saved as .pth files.

- **evaluation.py**: Evaluates the performance of trained CNN models on a test dataset, computes performance metrics, and generates a confusion matrix for each model.

- **requirements.txt**: Lists all the necessary Python libraries required to run the scripts effectively.

## Dependencies

- **argparse**: Utilized for parsing command-line options, arguments, and sub-commands.
- **opencv-python (cv2)**: A foundational library for image processing tasks, ranging from reading and writing images to complex manipulations.
- **mediapipe**: Utilized for its robust face detection capabilities, ensuring transformations are applied appropriately to face-centric images.
- **tqdm**: Provides a convenient progress bar for lengthy operations, improving the user interface and user experience during script execution.
- **Matplotlib & Seaborn**: These libraries are used for generating informative visualizations, such as class distributions and pixel intensity histograms, to analyze the dataset.
- **Pandas & NumPy**: Essential tools for data manipulation and numerical operations, enabling efficient handling and transformation of image data.
- **scikit-learn**: Offers a variety of machine learning algorithms and utilities for data preprocessing and model evaluation.
- **PyTorch & TorchVision**: Essential libraries for building and training neural networks.

## Datasets
- Datasets are accessible here (Login with a concordia email): https://liveconcordia-my.sharepoint.com/:f:/g/personal/lu_rami_live_concordia_ca/EvQYxf2VPVJHvL-TsQW0i9ABTrSW5WSeH4Pjz5NTlnD6Fg?e=LZXkmF

## Installation

Ensure Python is installed on your system. Install the required dependencies by navigating to the project's directory and running:

```sh
pip install -r requirements.txt

```
## Usage

The scripts are intended to be run from the command line. Here are some examples on how to run them, assuming you are in the project directory:

### Renaming Images

```
python renames_images.py <input_directory> <output_directory>
```

### Normalizing Images

To normalize images (`-c` flag is for color images. Default is greyscale.):

```
python normalize.py [-c for color images] <input_directory> <output_directory> 
```

### Augmenting Images

```
python augment.py <input_directory> <output_directory>
```
Note: The actual options for augmentation (such as specific transformations) should be configured within the script.

### Generating Plots

```
python generate_plots.py <image_directory>
```

### Analyzing Shuffled Images and RGB Distributions

```
python shuffle_plots.py <image_directory>
```

### Generating Training, Validation and Testing sets

```
python training.py
```
Note: The script generates three data loaders (data_loader.pt, valid_loader.pt, test_loader.pt) containing the training, validation, and testing sets respectively, ready for CNN training.

The script performs data augmentation, splitting the data into training, validation, and testing sets, and ensures the uniqueness of the image sets across the splits.


### Training CNN Models

```
python training.py
```

Note: Ensure that the necessary data loaders (data_loader.pt, valid_loader.pt, test_loader.pt) are available before running training.py.

### Evaluating CNN Models

```
python evaluation.py
```

Note: Ensure that the trained models and the test dataset are available, and the necessary data loaders are already generated before running the script.

It generates a confusion matrix for each model, visualizing the classification performance across different classes.
The script saves the performance metrics in a CSV file named evaluation_stats.csv.






