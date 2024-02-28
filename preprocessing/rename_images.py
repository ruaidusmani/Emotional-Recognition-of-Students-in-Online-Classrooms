import os
import cv2
import mediapipe as mp
import shutil
import argparse

def rename_images(inputDirectory, outputDirectory):

    # error handling for input directory
    if not os.path.exists(inputDirectory):
        print("\nInput directory does not exist.\n")
        return
    
    # initialize mediapipe
    mp_face_detection = mp.solutions.face_detection
    face_detect = mp_face_detection.FaceDetection()

    # function to rename multiple files

    folders_to_explore = ['focused', 'happy', 'neutral', 'surprised']

    # move up one directory
    print("current directory: ", os.getcwd())
    root_dir = os.path.dirname(os.getcwd())
    os.chdir('..')
    print(root_dir)
    os.chdir(inputDirectory)

    for folder in folders_to_explore:
        os.chdir(folder)
        print("BEFORE: ", os.getcwd())
        # checking if output directory already exists
        if not os.path.exists(f"{root_dir}/{outputDirectory}/{folder}"):
            os.makedirs(f"{root_dir}/{outputDirectory}/{folder}")
        # explore each subfolder
        img_count = 0 
        for subfolder in os.listdir():
            # Skip .DS_Store file
            if subfolder == '.DS_Store':
                continue
            # move into each subfolder
            os.chdir(subfolder)
            subfolder_name = os.path.basename(subfolder)
            # rename each file
            for count, filename in enumerate(os.listdir()):

                if img_count == 500:
                    break
                
                img = cv2.imread(filename)
                _, _, c = img.shape

                if c != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_detect.process(img)
                
                if not results.detections:
                    print("No face detected in ", filename)
                    continue
                
                img_count += 1

                # save the destination in one folder above
                dst = f"{root_dir}/{outputDirectory}/{folder}/{count}_{subfolder_name}.jpg"
                src = filename

                print(dst)
                # check if file exists
                if os.path.exists(dst):
                    # print("file exists")
                    continue
                else:
                    shutil.copy(src, dst)
            # move back to the parent folder
            print("current directory: ", os.getcwd())
            os.chdir('..')
            print("current directory: ", os.getcwd())
        os.chdir('..')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename images and move them to a new directory, also checking if a face is detected in the image.")
    parser.add_argument("inputDirectory", help="Directory containing images to rename (Directory has to be in the root directory of the project)")
    parser.add_argument("outputDirectory", help="Directory to move renamed images to")
    args = parser.parse_args()
    rename_images(args.inputDirectory, args.outputDirectory)