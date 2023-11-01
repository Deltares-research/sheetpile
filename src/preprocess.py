from PIL import Image
import os 
import numpy as np
import pickle
from pathlib import Path

IMAGE_SIZE = 256
IMAGE_IDX = 255
OUTPUT_CHANNELS = 1
BATCH_SIZE = 1


def load_all_data(directory):
    # get all the directories and load the data
    all_directories = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]
    all_data = []
    for directory in all_directories:
        print(f"Loading data from {directory}")
        # loop through the subdirectories
        for sub_directory in os.listdir(directory):
            print(f"Loading data from {directory}")
            # get all the png files
            directory = os.path.join(directory, sub_directory)
            # find png files using Pathlib
            all_files = list(Path(directory).glob('*.png'))
            # remove files that have stage_2 in the name
            all_files = [file for file in all_files if "stage_2" not in str(file)]
            feature_names = [file.stem for file in all_files]
            if len(all_files) >= 5:
                # load the data
                data = []
                for file in all_files:
                    print(f"Loading data from {file}")
                    image = Image.open(file)
                    # resize the image and maintain aspect ratio
                    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                    image_array = np.array(image)
                    # extract only the red channel since the image is grayscale
                    image_array = image_array[:, :, 0] / IMAGE_IDX
                    data.append(image_array)
                all_data.append(np.array(data))
    return feature_names, np.array(all_data).astype(np.float32)


if __name__ == '__main__':
    directory = f'final'
    # get all folders in output directory
    feature_names, all_data = load_all_data(directory)
    # dump the data into a pickle file
    data = {"feature_names": feature_names, "all_data": all_data}
    # create ml_input directory
    ml_input_directory = "final/ml_input"
    if not os.path.exists(ml_input_directory):
        os.makedirs(ml_input_directory)
    pickle.dump(data, open("final/ml_input/data.p", "wb"))

