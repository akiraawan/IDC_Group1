## code use for adding label to the files in kaggle data set
import os
import shutil


root_directory = "/Users/zilefeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Interdisplinary computing researching/database/testing"

export_directory = '/Users/zilefeng/Documents/GitHub/IDC_Group1/Dataset/testing'

count=1

for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    count+=1
    print(count)

    # Check if the item in the root directory is a folder
    if os.path.isdir(folder_path):
        # Construct the path to the info.cfg file within the current folder
        info_file_path = os.path.join(folder_path, 'info.cfg')

        # Check if the info.cfg file exists in the current folder
        if os.path.exists(info_file_path):
            # Open and read the contents of the info.cfg file
            with open(info_file_path, 'r') as info_file:
                info_contents = info_file.readlines()
                for line in info_contents:
                    if line.startswith("Group:"):
                        stat=line.strip("Group: ")
                        stat=stat.replace('\n', '') + '_'
                        print(stat)
                        break
                    
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith("4d.nii"):
                new_filename = stat+ filename
                export_path = os.path.join(export_directory, new_filename)
                shutil.copy(file_path, export_path)
                print(f"File '{filename}' has been exported as '{new_filename}' to '{export_directory}'.")


print("export done")
        