import os 

# function to rename multiple files

folders_to_explore = ['focused', 'happy', 'neutral', 'surprised']

# move up one directory
# print("current directory: ", os.getcwd())
root_dir = os.path.dirname(os.getcwd())
print(root_dir)
os.chdir('split_data')

for folder in folders_to_explore:
    current_category = os.path.basename(folder)
    os.chdir(folder)
    print("BEFORE: ", os.getcwd())
    # explore each subfolder 
    for subfolder in os.listdir():
        # move into each subfolder
        os.chdir(subfolder)
        subfolder_name = os.path.basename(subfolder)
        # rename each file
        for count, filename in enumerate(os.listdir()):
            # save the destination in one folder above
            dst = f"{root_dir}\COMP472\concat_data\{folder}\{count}_{subfolder_name}.jpg"
            src = filename
            dst = dst
            # check if file exists
            if os.path.exists(dst):
                # print("file exists")
                continue
            else:
                os.rename(src, dst)
        # move back to the parent folder
        print("current directory: ", os.getcwd())
        os.chdir('..')
        print("current directory: ", os.getcwd())
    os.chdir('..')