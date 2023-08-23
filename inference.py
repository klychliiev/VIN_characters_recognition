import cv2
import numpy as np
import os
import sys
from keras.api._v2.keras.models import load_model
from pathlib import Path

model = load_model('model.h5')

class_names = {
    '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F',
    '16': 'G', '17': 'H', '18': 'J', '19': 'K', '20': 'L', '21': 'M',
    '22': 'N', '23': 'P', '24': 'R', '25': 'S', '26': 'T', '27': 'U',
    '28': 'V', '29': 'W', '30': 'X', '31': 'Y', '32': 'Z'
}
project_folder = os.getcwd().split('/')[-1]

# define a function to process
def predict_image():

    # define a function to normalize the image (turn grayscale and reduce the format to 28*28 pixels)
    def normalize_image(image):
        img = cv2.imread(image, 0)
        resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        rescaled = resized / 255
        reshaped = np.array(rescaled).reshape(-1,
                                              rescaled.shape[0], rescaled.shape[1], 1)
        return reshaped

    # predict the character and convert it to decimal
    def predict(processed_image):
        predictions = model.predict(processed_image, verbose=0)
        prediction = str(np.argmax(predictions))
        prediction = class_names.get(prediction, prediction)
        ascii_prediction = ord(prediction)
        decimal_prediction = '{:03d}'.format(ascii_prediction)
        return decimal_prediction

    # process CLI arguments and print predictions for each image
    if (args_count := len(sys.argv)) > 2:
        print(f"One argument (folder path) expected, got {args_count - 1}")
        raise SystemExit(2)
    elif args_count < 2:
        print("You must specify the target directory")
        raise SystemExit(2)

    target_dir = Path(sys.argv[1])

    if not target_dir.is_dir():
        print("The target directory doesn't exist")
        raise SystemExit(1)

    for entry in target_dir.iterdir():
        image_path = os.path.join(str(target_dir), str(entry.name))
        entry = str(entry.name)
        if entry.endswith('.png') or entry.endswith('.jpg') or entry.endswith('.jpeg'):
            result = predict(normalize_image(image_path))
            path = os.path.join(project_folder, str(target_dir), entry)
            print(result + ', ' + path)


if __name__ == '__main__':
    predict_image()
