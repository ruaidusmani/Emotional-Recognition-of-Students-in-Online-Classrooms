import cv2
import os
import mediapipe as mp
import argparse
from tqdm import tqdm as tqdm_bar

# ----------------- Normalize Images -----------------
# This function takes in an input directory and an output directory 
# and normalizes all images in the input directory to 48x48 greyscale images.
# The function uses the mediapipe face detection model to detect the face in 
# the image and crops the image to the face.
# ---------------------------------------------------
def normalizeImages(inputDirectory, outputDirectory, color_option=False):

    # error handling for input directory
    if not os.path.exists(inputDirectory):
        print("\nInput directory does not exist.\n")
        return
    
    # initializing mediapipe for face detection
    mp_face_detect = mp.solutions.face_detection
    face_detect = mp_face_detect.FaceDetection()
    
    # Get a list of all the image files
    image_files = [os.path.join(root, file) for root, _, files in os.walk(inputDirectory) for file in files if file.endswith(".jpg")]

    for image_file in tqdm_bar(image_files, desc="Normalizing images", unit="images"):
        # using relative path to create same folder structure in output directory
        relative_path = os.path.relpath(os.path.dirname(image_file), inputDirectory)
        new_output_directory = os.path.join(outputDirectory, relative_path)
        os.makedirs(new_output_directory, exist_ok=True)
        
        # reading image
        img = cv2.imread(image_file)
        height, width, channel = img.shape

        # convert to rgb if grayscale
        if channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # convert to rgb since opencv reads in bgr but media pipe uses rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # find face in image
        results = face_detect.process(img)

        crop_size = 48
        cropped_face = None

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            height, width, _ = img.shape
            x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), int(bboxC.width * width), int(bboxC.height * height)

            # checking if area is square
            if w > h:
                margin = (w - h) // 2
                y -= margin
                h = w
            elif h > w:
                margin = (h - w) // 2
                x -= margin
                w = h

            # verifying that the bounding box is within the original image
            x, y, w, h = max(0, x), max(0, y), min(width - x, w), min(height - y, h)

            # cropping to square and resizing to 48x48 
            cropped_face = img[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

            # converting back to grayscale if output type is greyscale
            if color_option == False:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
            else:
                # reverting back to original color space
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(new_output_directory, os.path.basename(image_file)), cropped_face)

if __name__ == "__main__":
    # input parser for command line arguments
    # usage: python normalize.py inputDirectory outputDirectory
    parser = argparse.ArgumentParser(description='Normalize images to 48x48 greyscale')
    # option to output in rgb or greyscale. By default, greyscale is useds
    parser.add_argument('-c', action='store_true', help='output in rgb')
    parser.add_argument('inputDirectory', type=str, help='input directory')
    parser.add_argument('outputDirectory', type=str, help='output directory')
    args = parser.parse_args()
    color_option = True if args.c else False
    normalizeImages(args.inputDirectory, args.outputDirectory, args.c)
    

