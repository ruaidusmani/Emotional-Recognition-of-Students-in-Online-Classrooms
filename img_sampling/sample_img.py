import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import numpy as np

# Get current directory and go to it's parent directory
print(os.getcwd())
os.chdir("../")
print(os.listdir(os.getcwd()))

# Dictionary of images sorted by emotion
dict_images = {'focused': [], 'happy': [], 'neutral' : [], 'surprised': []}

# Attain all images from each emotion
for folder in os.listdir(os.getcwd()):
    print("Current folder: ", folder)
    if (folder == "Script"): # Skips the folder with script in it
        continue

    for image in os.listdir(folder): # Checks through directory to find .jpg files
        if image.endswith('.jpg'):
            img_path = os.path.abspath(os.path.join(folder, image))
            try:
                img = mpimg.imread(img_path)
                dict_images[folder].append(img_path) # Stores in the dictionary with specific emotion index
            except (OSError, IOError) as e:
                print(f"Error loading image: {img_path} - {e}")

# Plotting images in 5x5 grid    
rows = 5
columns = 5

# Plotting each emotion grid
for emotion in dict_images:
    # Shuffling the array and picking the first 25 images
    np.random.shuffle(dict_images[emotion])
    disp_img = dict_images[emotion][:25]

    fig = plt.figure(figsize=(10,10))
    
    for i, img_path in enumerate(disp_img):
       img = mpimg.imread(img_path)
       plt.subplot(rows, columns, i + 1)
       plt.imshow(img)
       plt.axis('off')

    plt.show()