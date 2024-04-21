import os
import cv2
import torch

def flatten_image(image_name, subfolder):
    img = cv2.imread("%s/"%(subfolder) + image_name, 0)
    return img


def get_image_subfolder(image_name):
    # expand folder ../concat_data_tagged
    for subfolder in os.listdir('../concat_data_tagged'):
        if image_name in os.listdir('../concat_data_tagged/' + subfolder):
            return subfolder
        
    


def get_race_category(image_name):
    #     tags = ['white', 'black', 'asian', 'hispanic', 'other_race']
    if '_white' in image_name:
        return 'white'
    elif '_black' in image_name:
        return 'black'
    elif '_asian' in image_name:
        return 'asian'
    elif '_hispanic' in image_name:
        return 'hispanic'
    elif '_other_race' in image_name:
        return 'other_race'
    else:
        print('Error: Could not determine the race category of the image: ', image_name)
        return None
def get_age_category(image_name):
        # elif group == 'age':
    #     tags = ['young', 'adult', 'old']

    if '_young' in image_name:
        return 'young'
    elif '_adult' in image_name:
        return 'adult'
    elif '_old' in image_name:
        return 'old'
    else:
        print('Error: Could not determine the age category of the image: ', image_name)
        return None
def get_gender_category(image_name):
        # elif group == 'gender':
    #     tags = ['male', 'female', 'other_gender']
    if '_male' in image_name:
        return 'male'
    elif '_female' in image_name:
        return 'female'
    elif '_other_gender' in image_name:
        return 'other_gender'
    else:
        print('Error: Could not determine the gender category of the image: ', image_name)
        return None


def test_individual_image(model, image_name, category, labels, read_custom_path = ''):
    if (read_custom_path != ''):
        img = cv2.imread(read_custom_path, 0)
        image_name = read_custom_path
        #save the image
    else:
        img = cv2.imread("../concat_data_tagged/%s/%s" % (category, image_name), 0)
    
    #resize to 48x48 pixels
    img = cv2.resize(img, (48, 48))



    #if image 3 channel convert to 1 channel
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("image_mod.jpg", img)
    
    # Convert image to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Get label tensor
    

    label_tensor = torch.tensor(labels[category], dtype=torch.long)

    # Forward pass for the single image
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    # Print results
    # print("Predicted:", predicted.item(), "(", reverse_labels[predicted.item()], ")")
    # print("Actual:", label_tensor.item(), "(", reverse_labels[label_tensor.item()], ")")
    # print("Image:", image_name)
    # print("Category:", category)
    # print("///////")
    
    return predicted.item()

