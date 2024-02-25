import os
import shutil

directory = r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\CFD"

#go through each folder in the directory
for folder in os.listdir(directory):
    #check if it's a folder
    if os.path.isdir(directory + "\\" + folder):
        #check if the extension is a .jpg extension
        # if it is, save the file name to another directory
        for file in os.listdir(directory + "\\" + folder):
            #check if filename contains "-N"
            if "-H" in file:
                if file.endswith(".jpg"):
                    
                    #don't move the file, copy the file to another directory

                    #copy the file to another directory
                    shutil.copy(directory + "\\" + folder + "\\" + file, r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\all_images\happy\\" + file)
                    
                    # os.rename(directory + "\\" + folder + "\\" + file, r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\all_images\\" + file)


directory = r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\CFD-INDIA"

#go through each file in the directory
for file in os.listdir(directory):
    #check if it's a folder
    if not os.path.isdir(directory + "\\" + folder):
        #check if the extension is a .jpg extension
        # if it is, save the file name to another directory
        #check if filename contains "-N"
        if "-H" in file:
            if file.endswith(".jpg"):
                
                #don't move the file, copy the file to another directory

                #copy the file to another directory
                shutil.copy(directory + "\\" + file, r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\all_images\happy\\" + file)                
                # os.rename(directory + "\\" + folder + "\\" + file, r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\all_images\\" + file)


directory = r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\CFD-MR"

#go through each file in the directory
for file in os.listdir(directory):
    #check if it's a folder
    if not os.path.isdir(directory + "\\" + folder):
        #check if the extension is a .jpg extension
        # if it is, save the file name to another directory
        #check if filename contains "-N"
        if "-H" in file:
            if file.endswith(".jpg"):
                
                #don't move the file, copy the file to another directory

                #copy the file to another directory
                shutil.copy(directory + "\\" + file, r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\all_images\happy\\" + file)                
                # os.rename(directory + "\\" + folder + "\\" + file, r"C:\Users\Luis\Downloads\chicago_faces\cfd30norms\CFD Version 3.0\Images\all_images\\" + file)




                    