import os
import shutil
import PIL
from PIL import Image
import random

def MinDictCount(path):
    dict = {}
    img_count = []
    for idx, (root, dirs, files) in enumerate(os.walk(path, topdown=True)):
        if idx != 0:
            # Folder name
            dict[root.split('\\')[-1]] = {}
            # Image list
            dict[root.split('\\')[-1]]['image_list'] = os.listdir(root)
            # Number of images in this folder
            dict[root.split('\\')[-1]]['image_count'] = len(os.listdir(root))
            img_count.append(len(os.listdir(root)))
    return dict, max(img_count)

def CV_Count(min, dict, trainRate=0.8, valRate=0.1):
    test_rate = 1 - trainRate - valRate
    for items in dict:
        total_images = dict[items]['image_count']
        dict[items]['Train'] = round(total_images * trainRate)
        dict[items]['Val'] = round(total_images * valRate)
        dict[items]['Test'] = total_images - dict[items]['Train'] - dict[items]['Val']

    return dict

def CreateFold(fold_num, dict, source_path, dest_path):
    data_path = os.path.join(dest_path, 'Data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    folder_names = ['Train', 'Val', 'Test']
    for name in folder_names:
        folder_path = os.path.join(data_path, name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for id in range(1, fold_num + 1):
            fold_path = os.path.join(folder_path, 'fold_' + str(id))
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)

def CrossValidation(dict, path, dest_path, fold_num,
                    resize=(124, 124),
                    random_state=True,
                    video=False,
                    nifti=False,
                    Images=True,
                    image_format='.jpg',
                    audio=False,
                    audio_file_format='.wav'):
    for classes in dict:
        source_folder_path = os.path.join(path, classes)
        split = round(dict[classes]['image_count'] / fold_num)
        print(classes)
        copy_image_list = dict[classes]['image_list'].copy()
        if random_state:
            copy_image_list = random.sample(copy_image_list, len(copy_image_list))
        image_list = copy_image_list + copy_image_list

        for folds in range(0, fold_num):
            # Training set
            for images in image_list[(0 + (folds * split)): (dict[classes]['Train'] + (folds * split))]:
                image_path = os.path.join(source_folder_path, images)
                dest_folder_path = os.path.join(dest_path, 'Data', 'Train', 'fold_' + str(folds + 1), classes)
                if not os.path.exists(dest_folder_path):
                    os.mkdir(dest_folder_path)
                if Images:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + image_format)
                    image = Image.open(image_path)
                    try:
                        image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
                        image = image.convert('RGB')
                        image.save(dest_image_path)
                    except OSError:
                        print(image_path)
                        continue

            # Validation set
            val_start = dict[classes]['Train'] + (folds * split)
            val_end = val_start + dict[classes]['Val']
            for images in image_list[val_start: val_end]:
                image_path = os.path.join(source_folder_path, images)
                dest_folder_path = os.path.join(dest_path, 'Data', 'Val', 'fold_' + str(folds + 1), classes)
                if not os.path.exists(dest_folder_path):
                    os.mkdir(dest_folder_path)
                if Images:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + image_format)
                    image = Image.open(image_path)
                    try:
                        image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
                        image = image.convert('RGB')
                        image.save(dest_image_path)
                    except OSError:
                        print(image_path)
                        continue

            # Test set
            test_start = val_end
            test_end = test_start + dict[classes]['Test']
            for images in image_list[test_start: test_end]:
                image_path = os.path.join(source_folder_path, images)
                dest_folder_path = os.path.join(dest_path, 'Data', 'Test', 'fold_' + str(folds + 1), classes)
                if not os.path.exists(dest_folder_path):
                    os.mkdir(dest_folder_path)
                if Images:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + image_format)
                    image = Image.open(image_path)
                    try:
                        image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
                        image = image.convert('RGB')
                        image.save(dest_image_path)
                    except OSError:
                        print(image_path)
                        continue

def main():
    # Path of the Source folder
    path = r"D:/MN/GastroVision2/Gastrovision/Gastrovision"

    # Path of the destination folder
    dest_path = r"D:/MN/gastro three stage/SaveFileForStage3/fold"

    # Creating a dictionary with folder name, image list, train, test, val counts
    dict, max = MinDictCount(path)

    # Specify the trainRate and valRate
    dict = CV_Count(min, dict, trainRate=0.8, valRate=0.1)

    # Cross fold number set to 5, can be adjusted as needed
    CreateFold(5, dict, path, dest_path)

    # Cross-validation setup
    CrossValidation(dict, path, dest_path, 5, resize=(124, 124),
                    random_state=False, video=False, nifti=False,
                    Images=True,
                    image_format='.png',
                    audio=False,
                    audio_file_format='.wav'
                    )

if __name__ == "__main__":
    main()
