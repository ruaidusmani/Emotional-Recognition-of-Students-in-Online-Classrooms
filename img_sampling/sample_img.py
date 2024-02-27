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



# %%
# %matplotlib notebook
# %matplotlib tk
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv

# %%
#sample 25 random images

print(os.getcwd())
os.chdir("../concat_data")

#dictionary that stores the file names of the images that were sampled
dict_images = {'focused': [], 'happy': [], 'neutral' : [], 'surprised': []}

for folder in os.listdir(os.getcwd()):

    num_dict = {}
    #get all file_names
    count_images = 0
    for image in os.listdir(folder):
        if (image.endswith('.jpg')):
            num_dict[count_images] = str(image)
            count_images += 1

    #roll a RNG 25 times and generate a number between
    #0 and the number of images in the folder
    num_already_rolled = []
    count = 0
    if (count_images > 0) :
        while count < 25:
            rand_num = np.random.randint(0, count_images)
            if (rand_num not in num_already_rolled):
                
                dict_images[folder].append(num_dict[rand_num])
                count += 1
             

# %%
#grab each category and append it to a list
dict_images_values = {'focused': {}, 'happy': {}, 'neutral' : {}, 'surprised': {}}

for folder in os.listdir(os.getcwd()):

    red_pixel_arr = []
    green_pixel_arr = []
    blue_pixel_arr = []

    for image in dict_images[folder]:
        #read test.jpg
        # img_array = cv.imread('../preprocessing/test.jpg')
        img_array = cv.imread(folder + "/" + image)
        #convert to rgb
        img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
        img_array_1D = img_array.flatten()

        #get each R, G and B value and add to an array

        #grab each pixel
        i = 0
        while i < len(img_array_1D):
            red_pixel_arr.append(img_array_1D[i])
            green_pixel_arr.append(img_array_1D[i+1])
            blue_pixel_arr.append(img_array_1D[i+2])
            i += 3
    dict_images_values[folder]['red'] = red_pixel_arr
    dict_images_values[folder]['green'] = green_pixel_arr
    dict_images_values[folder]['blue'] = blue_pixel_arr
    




# %%
print(len(dict_images_values['neutral']['red']))
# %%
#make 3x4 plot for each folder
fig, axs = plt.subplots(4, 3)
fig.set_size_inches(21, 10.5)
fig.suptitle('RGB Histograms')

i = 0
for key in dict_images_values:
# for key in ['neutral']:
    #add title to each row

    axs[i, 0].hist(dict_images_values[key]['red'], bins=255, color='red')
    axs[i, 1].hist(dict_images_values[key]['green'], bins=255, color='green')
    axs[i, 2].hist(dict_images_values[key]['blue'], bins=255, color='blue')
    axs[i, 0].set_title(key + " Red")
    axs[i, 1].set_title(key + " Green")
    axs[i, 2].set_title(key + " Blue")
    axs[i, 0].set_xlim([0, 255])
    axs[i, 1].set_xlim([0, 255])
    axs[i, 2].set_xlim([0, 255])
    

    i =+ 1
        
plt.show()

# %%



