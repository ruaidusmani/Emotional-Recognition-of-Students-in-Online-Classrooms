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
def normalizeImages(inputDirectory, outputDirectory):
    
    # initializing mediapipe for face detection
    mp_face_detect = mp.solutions.face_detection
    face_detect = mp_face_detect.FaceDetection()

    for root, _, files in tqdm_bar(os.walk(inputDirectory), desc="Normalizing images", unit=" images"):
        for file in files:
            if file.endswith(".jpg"):
                # using relative path to create same folder structure in output directory
                relative_path = os.path.relpath(root, inputDirectory)
                new_output_directory = os.path.join(outputDirectory, relative_path)
                os.makedirs(new_output_directory, exist_ok=True)
                
                # reading image
                img = cv2.imread(os.path.join(root, file))
                height, width, channel = img.shape

                # convert to rgb if grayscale
                if channel == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # convert to rgb if bgr
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
                    cropped_face = cv2.resize(cropped_face, (crop_size, crop_size))

                    # converting back to grayscale
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

                    cv2.imwrite(os.path.join(new_output_directory, file), cropped_face)
        
if __name__ == "__main__":
    # input parser for command line arguments
    # usage: python normalize.py inputDirectory outputDirectory
    parser = argparse.ArgumentParser(description='Normalize images to 48x48 greyscale')
    parser.add_argument('inputDirectory', type=str, help='input directory')
    parser.add_argument('outputDirectory', type=str, help='output directory')
    args = parser.parse_args()
    normalizeImages(args.inputDirectory, args.outputDirectory)

