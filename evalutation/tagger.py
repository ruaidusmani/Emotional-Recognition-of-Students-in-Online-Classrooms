# this code will run a tagger on the different datasets to assign
# a race, gender and age to each user.

# for gender: male, female, other
# for race: white, black, asian, hispanic, other
# for age: young (0-25), middle (26-50), old (51-100)

import os
import cv2


def tag_category(folder, category):

    # Look through images in concat_data_tagged folder
    # and all the subfolders (focused, happy, surprised, neutral)
    subfolders = folder

    #change current working directory

    print(os.getcwd())
    # for each subfolder
    for subfolder in subfolders:
        # get the list of images in the subfolder
        images = os.listdir('concat_data_tagged/' + subfolder)
        # for each image
        current_count = 0
        for image in images:
            print ('Processing image ' + str(current_count) + ' of ' + str(len(images)))
            image_path = 'concat_data_tagged/' + subfolder + '/' + image
            # prompt the user to add a specific tag 
            if (not check_if_image_has_categories_assigned(category, image)):
                print(image_path)
                tag = assign_categories(category, image_path)
                new_image_name = str(image+'_'+tag).replace('.jpg', '') + '.jpg'
                os.rename(image_path, 'concat_data_tagged/' + subfolder + '/' + new_image_name)
            current_count  = current_count + 1
                

def check_if_image_has_categories_assigned(group, image_name):
    if group == 'race':
        if 'white' in image_name or 'black' in image_name or 'asian' in image_name or 'hispanic' in image_name or 'other_race' in image_name:
            return True
        else:
            return False
    elif group == 'age':
        if 'young' in image_name or 'adult' in image_name or 'old' in image_name:
            return True
        else:
            return False
    elif group == 'gender':
        if 'male' in image_name or 'female' in image_name or 'other_gender' in image_name:
            return True
    else:
        return False

def assign_categories(group, image_name):
    
    if group == 'race':
        tags = ['white', 'black', 'asian', 'hispanic', 'other_race']
    elif group == 'age':
        tags = ['young', 'adult', 'old']
    elif group == 'gender':
        tags = ['male', 'female', 'other_gender']

    image = cv2.imread(image_name)
    image = cv2.resize(image, (250, 250))
    cv2.imshow("Image sample",image)
    #wait for image to be displayed
    cv2.waitKey(10)
    #press keystroke to close the image
    
    valid_input = False
    while not valid_input:
        for i in range(len(tags)):
            print(str(i) + ': ' + tags[i])
        category = input('Please assign a category to the image: ')
        if category.isdigit() and int(category) < len(tags):
            valid_input = True
        else:
            print('Invalid input. Please try again.')
    cv2.destroyAllWindows()
    return tags[int(category)]

def confirm_all_are_tagged():
    all_images = os.listdir('concat_data_tagged/')
    for subfolder in ['focused', 'happy', 'surprised', 'neutral']:
        images = os.listdir('concat_data_tagged/' + subfolder)
        count_tagged = 0
        for image in images:
            if (not check_if_image_has_categories_assigned):
                print('Image ' + image + ' has not been tagged.')
            else:
                count_tagged = count_tagged + 1
        print('There is ' + str(count_tagged) + ' tagged images in ' + subfolder + ' folder.')
#define main
if __name__ == '__main__':
    # tag_category(['focused', 'happy', 'surprised', 'neutral
    os.chdir('../')        
    confirm_all_are_tagged()
    tag_category(['focused'], 'age')