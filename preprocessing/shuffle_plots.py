import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import numpy as np
import cv2 as cv

# Get current directory and go to it's parent directory
print(os.getcwd())
os.chdir("concat_data")
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
            dict_images[folder].append(str(image)) # Stores in the dictionary with specific emotion index

for key in dict_images:
    print(key)
    print(len(dict_images[key]))

# Plotting images in 5x5 grid    
rows = 5
columns = 5

# Plotting each emotion grid
for emotion in dict_images:
    # Shuffling the array and picking the first 25 images
    np.random.shuffle(dict_images[emotion])
    disp_img = dict_images[emotion][:25]

    fig = plt.figure(figsize=(10,10))
    
    for i, file_name in enumerate(disp_img):
       img = mpimg.imread(emotion + "/" + file_name)
       plt.subplot(rows, columns, i + 1)
       plt.imshow(img)
       plt.axis('off')

    plt.show()
    plt.savefig('../preprocessing/5x5_grid_%s.png'%(emotion))



#grab each category and append it to a list
dict_images_values = {'focused': {}, 'happy': {}, 'neutral' : {}, 'surprised': {}}

for folder in os.listdir(os.getcwd()):

    #generate a list for each R, G, and B value
    red_pixel_arr = []
    green_pixel_arr = []
    blue_pixel_arr = []

    # iterage through the 25 selected images
    for image in dict_images[folder]:

        img_array = cv.imread(folder + "/" + image)
        
        #convert to rgb
        img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)

        #flatten the 2D array into a 1 dimensional array
        #it will contain a pixel intensity array for
        # each R, G, and B value like so: [R, G, B, R, G, B, ...]

        img_array_1D = img_array.flatten()


        # iterate through the array and append to the respective array
        i = 0
        while i < len(img_array_1D):
            red_pixel_arr.append(img_array_1D[i])
            green_pixel_arr.append(img_array_1D[i+1])
            blue_pixel_arr.append(img_array_1D[i+2])
            i += 3

            #failsafe for unprocessed images
            if (len(red_pixel_arr) > 1000000):
                break

    # append the arrays to the dictionary
    dict_images_values[folder]['red'] = red_pixel_arr
    dict_images_values[folder]['green'] = green_pixel_arr
    dict_images_values[folder]['blue'] = blue_pixel_arr


#make 3x4 plot for each folder
fig, axs = plt.subplots(4, 3)
fig.set_size_inches(21, 15)
fig.suptitle('RGB Histograms')

i = 0
for key in dict_images_values:
    #add title to each row
    print("generating graph for key: " + key)

    # set histogram values
    axs[i, 0].hist(dict_images_values[key]['red'], bins=255, color='red')
    axs[i, 1].hist(dict_images_values[key]['green'], bins=255, color='green')
    axs[i, 2].hist(dict_images_values[key]['blue'], bins=255, color='blue')

    # set histogram titles
    axs[i, 0].set_title(key + " Red")
    axs[i, 1].set_title(key + " Green")
    axs[i, 2].set_title(key + " Blue")

    #set histogram x-axis limits
    axs[i, 0].set_xlim([0, 255])
    axs[i, 1].set_xlim([0, 255])
    axs[i, 2].set_xlim([0, 255])
    

    i =+ 1
        
#Export plot as a png
plt.savefig('../preprocessing/histogram.png')






