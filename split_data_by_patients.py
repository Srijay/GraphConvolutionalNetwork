import os, shutil

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

input_patches_dir = r"F:\Datasets\RA\dawood\data\patches"
split_dir = r"F:\Datasets\RA\dawood\data\split"
train_split_dir = os.path.join(split_dir, "train")
test_split_dir = os.path.join(split_dir, "test")
mkdir(train_split_dir)
mkdir(test_split_dir)

train_patients = ['A', 'B', 'C', 'D', 'E']
test_patients = ['F', 'G', 'H']

image_names = os.listdir(input_patches_dir)

for imname in image_names:
    patient_id = imname.split("_")[0][0]
    source_file = os.path.join(input_patches_dir, imname)
    if(patient_id in train_patients):
        shutil.copy(source_file, os.path.join(train_split_dir,imname))
    elif(patient_id in test_patients):
        shutil.copy(source_file, os.path.join(test_split_dir, imname))
    else:
        print("Unidentified patient id ", patient_id)
        exit()
