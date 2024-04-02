# %%
import numpy as np
import cv2
import os
import sys
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import f1_score

# %%
# Hyperparameters and settings
batch_size = 64
test_batch_size = 64
input_size = 1 # because there is only one channel 
output_size = 4
num_epochs = 1000
learning_rate = 0.001



# %%
# Load traiing, validation and training data

data_loader = torch.load('data_loader.pt')
valid_loader = torch.load('valid_loader.pt')
test_loader = torch.load('test_loader.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the data loaders to the desired device
data_loader.dataset.tensors = tuple(tensor.to(device) for tensor in data_loader.dataset.tensors)
valid_loader.dataset.tensors = tuple(tensor.to(device) for tensor in valid_loader.dataset.tensors)
test_loader.dataset.tensors = tuple(tensor.to(device) for tensor in test_loader.dataset.tensors)


# %%
class CNN(nn.Module):
    def __init__(self):
        self.name = "CNN"
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(12 * 12 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # print(x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # fc layer
        x = self.fc_layer(x)
        return x



# %%
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.name = "CNN2"
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #get dimensions of last layer
            
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64*9*9, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # print(x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # fc layer
        x = self.fc_layer(x)
        return x

# %%
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.name = "CNN3"
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(7 * 7 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 4)
        )
        
    def forward(self, x, y=None):
        # conv layers
        x = self.conv_layer(x)
        # print(x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # fc layer
        x = self.fc_layer(x)
        return x

# %%
def train_and_save_model(model, data_loader, valid_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(data_loader)
    loss_list = []
    acc_list = []
    f1_list = []

    loss_list_test = []
    acc_list_test = []
    f1_list_test = []

    # early stopping parameters
    best_loss = 100
    patience = 10 # number of epochs to wait before stopping
    count = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):

            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            # Train F1 score
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
            f1_list.append(f1)
            # print(i)
            # if (i + 1) % 10 == 0:
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            #     .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
            #     (correct / total) * 100))

        with torch.no_grad():
            for i, (images_test, labels_test) in enumerate(valid_loader):
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)
                outputs_test = model(images_test)
                loss_test = criterion(outputs_test, labels_test)
                loss_list_test.append(loss_test.item())
                total_test = labels_test.size(0)
                _, predicted_test = torch.max(outputs_test.data, 1)
                correct_test = (predicted_test == labels_test).sum().item()
                acc_list_test.append(correct_test / total_test)

                #Test F1 score
                f1_test = f1_score(labels_test.cpu().numpy(), predicted_test.cpu().numpy(), average='macro')
                f1_list_test.append(f1_test)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, F1 Score: {:.2f}'
        .format(epoch + 1, num_epochs, i + 1, total_step, loss_test.item(),
        (correct / total) * 100, f1_test))


        #================= Early stopping =================
        
        #---------- Early stopping with avg F1 score-------
        # early stopping to prevent overfitting
        # saving model with best accuracy
        # avg_f1 = sum(f1_list_test)/len(f1_list_test)
        # print('Average F1 score: ', avg_f1)
        # if avg_f1 > best_f1:
        #     best_f1 = avg_f1
        #     count = 0 # reset count
        #     #saving best model thus far
        #     torch.save(model.state_dict(), '%s.pth'%(model.name))
        # else:
        #     count += 1
        #     if count == patience:
        #         print('Early stopping at epoch %d'%(epoch))
        #         break

        #------------- Early stopping with loss------------
        if loss_test.item() < best_loss:
            best_loss = loss_test.item()
            count = 0 # reset count
            best_epoch = epoch
            #saving best model thus far
            torch.save(model.state_dict(), '%s.pth'%(model.name))
            if best_loss < 0.0001:
                print('\n----------------------------------------\nEarly stopping at epoch %d'%(epoch))
                print('Best epoch: ', best_epoch)
                print('Best loss: ', best_loss, ' Best test F1 score: ', f1_test, ' Best test accuracy: ', acc_list_test[-1])
                break
        else:
            count += 1
            if count == patience:
                print('\n----------------------------------------\nEarly stopping at epoch %d'%(epoch))
                print('Best epoch: ', best_epoch)
                print('Best loss: ', best_loss, ' Best test F1 score: ', f1_test, ' Best test accuracy: ', acc_list_test[-1])
                break

        



# %%
model1 = CNN().to(device)
train_and_save_model(model1, data_loader, valid_loader, device)


# %%
#device = torch.device('cpu')

model2 = CNN2().to(device)

train_and_save_model(model2, data_loader, valid_loader, device)


# %%

model3 = CNN3().to(device)
train_and_save_model(model3, data_loader, valid_loader, device)

# %%
def evaluate_model(model, valid_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        print('Test Accuracy of the model (validation): {:.2f} %'
        .format((correct / total) * 100))
        print('F1 Score of the model (validation): {:.2f} %'
        .format((f1_score(all_labels, all_predictions, average='macro')) * 100))

def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        print('Test Accuracy of the model (testing): {:.2f} %'
        .format((correct / total) * 100))
        print('F1 Score of the model (testing): {:.2f} %'
        .format((f1_score(all_labels, all_predictions, average='macro')) * 100))

# %%


# %%
for model in [model1, model2, model3]:
    print('\nModel: ', model.name, '\n')

    evaluate_model(model, valid_loader)
    print('\n')
    test_model(model, test_loader)

# %%


# %%
# function to test an individual image from an external source
def test_individual_image(model, image_name, category, read_custom_path = ''):
    if (read_custom_path != ''):
        img = cv2.imread(read_custom_path, 0)
        image_name = read_custom_path
        #save the image
    else:
        img = cv2.imread("../concat_data/%s/%s" % (category, image_name), 0)
    
    #resize to 48x48 pixels
    img = cv2.resize(img, (48, 48))



    #if image 3 channel convert to 1 channel
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("image_mod.jpg", img)
    
    # Convert image to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Get label tensor
    labels = {'focused': 0, 'happy': 1, 'neutral': 2, 'surprised': 3}
    reverse_labels = {0: 'focused', 1: 'happy', 2: 'neutral', 3: 'surprised'}
    label_tensor = torch.tensor(labels[category], dtype=torch.long)

    # Forward pass for the single image
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    # Print results
    print("Predicted:", predicted.item(), "(", reverse_labels[predicted.item()], ")")
    print("Actual:", label_tensor.item(), "(", reverse_labels[label_tensor.item()], ")")
    print("Image:", image_name)
    print("Category:", category)
    print("///////")
    
    return predicted.item()

# %%
test_individual_image(model2, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "happy", read_custom_path=r"../unseen-test-imgs/test_smile.PNG")


# %%
test_individual_image(model3, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "neutral", read_custom_path=r"../unseen-test-imgs/test_neutral.PNG")

# %%
test_individual_image(model2, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "focused", read_custom_path=r"../unseen-test-imgs/test_focused.PNG")
test_individual_image(model2, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "focused", read_custom_path=r"../unseen-test-imgs/test_focused_2.PNG")
test_individual_image(model2, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "focused", read_custom_path=r"../unseen-test-imgs/test_focused_3.PNG")
test_individual_image(model2, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "focused", read_custom_path=r"../unseen-test-imgs/test_focused_4.PNG")
test_individual_image(model2, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "focused", read_custom_path=r"../unseen-test-imgs/test_focused_5.PNG")

# %%
test_individual_image(model1, "86_MMA-FACIAL-EXPRESSION-mahmoud.jpg", "surprised", read_custom_path=r"../unseen-test-imgs/test_surprised.PNG")


