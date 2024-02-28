import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# ----------------- Augment Images -----------------
# This function takes in an input directory and an output directory
# and augments all images in the input directory and saves them in the output directory.
# The function flips, rotates, brightens, and changes the contrast of the images.
# ---------------------------------------------------
def augment(inputDirectory, outputDirectory):

    # error handling for input directory
    if not os.path.exists(inputDirectory):
        print("\nInput directory does not exist.\n")
        return
    
    # Get a list of all the image files
    image_files = [os.path.join(root, file) for root, _, files in os.walk(inputDirectory) for file in files if file.endswith(".jpg")]

    for image_file in tqdm(image_files, desc="Augmenting images: ", unit="images"):
        # using relative path to create same folder structure in output directory
        relative_path = os.path.relpath(os.path.dirname(image_file), inputDirectory)
        new_output_directory = os.path.join(outputDirectory, relative_path)
        os.makedirs(new_output_directory, exist_ok=True)
        
        # reading image
        img = cv2.imread(image_file)

        file_name = os.path.basename(image_file)

        # save original image in new directory
        cv2.imwrite(os.path.join(new_output_directory, file_name), img)
        
        # ---- Augmenting images ----

        # flipping
        cv2.imwrite(os.path.join(new_output_directory, "flipped_" + file_name), flip(img))
        
        # rotating
        cv2.imwrite(os.path.join(new_output_directory, "rotated_" + file_name), rotate(img))

        # brightening
        cv2.imwrite(os.path.join(new_output_directory, "brightened_" + file_name), brighten(img))

        # changing contrast
        cv2.imwrite(os.path.join(new_output_directory, "contrast_" + file_name), change_contrast(img))

        # ---------------------------
        

def flip(image):
    return cv2.flip(image, 1) # flipping around y-axis

def rotate(image):
    angle = np.random.uniform(-5,5)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def brighten(image):
    factor = np.random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def change_contrast(image):
    factor = np.random.uniform(0.5, 1.5)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

if __name__ == "__main__":
    # ex usage: python augment.py ../data/processed/ ../data/augmented/
    parser = argparse.ArgumentParser(description="Augment images and move them to a new directory.")
    parser.add_argument("inputDirectory", help="The directory containing the images to be augmented.")
    parser.add_argument("outputDirectory", help="The directory to move the augmented images to.")
    args = parser.parse_args()
    augment(args.inputDirectory, args.outputDirectory)