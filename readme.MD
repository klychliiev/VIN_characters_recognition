# **VIN characters recognition using Python and Deep Learning**

This project is aimed at VIN characters recognition. Full notebook describing data preparation process, model training and model evaluation can be found at https://colab.research.google.com/drive/17AjHy4JntLBbtQYrEFTqc8uGMuNmQNHy

---

## Getting started with the project

To run the program two folders are needed: the first is the unzipped project folder and the second is the test data (scanned images), which one should place inside the first folder.
Move to the project folder (ml_internship_project). Below is the example of program output using requirements.txt:

```bash
ls 
images  inference.py  model.h5  readme.MD  requirements.txt  train.py

ls images
4.png  5.png  f.jpg  f.png  l.png  n.png  p.png  t.png  v.png  w.png  z.png

python3 -m venv myvenv

source myvenv/bin/activate

pip3 install -r requirements.txt

python3 inference.py images
090, vin_recognition/images/z.png
052, vin_recognition/images/4.png
087, vin_recognition/images/n.png
084, vin_recognition/images/t.png
076, vin_recognition/images/l.png
087, vin_recognition/images/w.png
070, vin_recognition/images/f.jpg
080, vin_recognition/images/p.png
086, vin_recognition/images/v.png
049, vin_recognition/images/5.png
070, vin_recognition/images/f.png
```

Note, after running the program you receive the output as follow:
090, ml_internship_project/images/z.png,
where the first argument, 090, is an ASCII code for recognized character (090 is Z) and the second argument is the directory to the image. As you can see from the file name, the character on the image is also Z, so our model predicted it correctly.

## Python packages used in this project

- cv2 for image processing
- numpy for array computations
- sklearn for model evaluation report
- matplotlib, seaborn for data visualization
- pandas for tabular data processing
- keras for deep learning
- emnist for obtaining handwritten data

## Dataset

The dataset used in this project is The EMNIST Dataset, a set of handwritten digits and letters converted to a 28*28 pixel image format. More information about the dataset can be found via https://www.nist.gov/itl/products-and-services/emnist-dataset. emnist Python package was used to obtain the data. The package provides functionality to automatically download and cache the dataset, and to load it as numpy arrays. After preprocessing our train set contains 4800 images of each character (158400 in total), and test set contains 800 (26400 in total).

## Classification model

The model developed for multi-class image classification was built using CNN, which proved to be effective when working with image and video data. The neural network receives as an input arrays representing 28*28 grayscale images. It consists of 2 convolutional layers, one pooling layer and 2 fully connected layers. The multi-class problem classes are exclusive, so the softmax activation function is used in the output layer, which consists of 33 neurons, corresponding to the number of classes (10 digits + 23 letters).

**Model architecture**:

Model: "sequential"

---

# Layer (type)                Output Shape              Param

conv2d (Conv2D)             (None, 24, 24, 128)       3328

max_pooling2d (MaxPooling2D  (None, 12, 12, 128)      0  
)

dropout (Dropout)           (None, 12, 12, 128)       0

conv2d_1 (Conv2D)           (None, 8, 8, 256)         819456

max_pooling2d_1 (MaxPooling  (None, 4, 4, 256)        0  
2D)

dropout_1 (Dropout)         (None, 4, 4, 256)         0

flatten (Flatten)           (None, 4096)              0

dropout_2 (Dropout)         (None, 4096)              0

dense (Dense)               (None, 1024)              4195328

dropout_3 (Dropout)         (None, 1024)              0

dense_1 (Dense)             (None, 33)                33825

=================================================================
Total params: 5,051,937
Trainable params: 5,051,937
Non-trainable params: 0

---

Another model architecture was used to solve this problem. It showed a bit better performance, but it overfits, as shown on the graph.

![graph](metadata/graph.jpg)

## Model performance

The model tested on unseen data showed an overall accuracy of 93%.

The classification report generated using sklearn shows good results for most characters (over 0.9). The execptions are '1' and 'L' which are often misclassified due to the fact, that The EMNIST Dataset also contains images of handwritten lowercase 'L', which resembles '1' a lot.

The classification report:

           0       0.96      0.97      0.97       800
           1       0.69      0.74      0.71       800
           2       0.91      0.92      0.92       800
           3       0.98      0.98      0.98       800
           4       0.95      0.93      0.94       800
           5       0.92      0.90      0.91       800
           6       0.94      0.96      0.95       800
           7       0.98      0.99      0.98       800
           8       0.97      0.96      0.96       800
           9       0.88      0.95      0.91       800
           A       0.94      0.96      0.95       800
           B       0.95      0.93      0.94       800
           C       0.97      0.96      0.97       800
           D       0.95      0.95      0.95       800
           E       0.96      0.97      0.97       800
           F       0.98      0.96      0.97       800
           G       0.89      0.78      0.83       800
           H       0.94      0.96      0.95       800
           J       0.97      0.97      0.97       800
           K       0.98      0.95      0.97       800
           L       0.69      0.67      0.68       800
           M       0.98      0.99      0.98       800
           N       0.96      0.96      0.96       800
           P       0.98      0.99      0.98       800
           R       0.96      0.95      0.96       800
           S       0.90      0.92      0.91       800
           T       0.96      0.98      0.97       800
           U       0.95      0.94      0.94       800
           V       0.94      0.94      0.94       800
           W       0.99      0.98      0.99       800
           X       0.98      0.96      0.97       800
           Y       0.91      0.93      0.92       800
           Z       0.92      0.92      0.92       800
    
    accuracy                           0.93     26400

macro avg       0.93      0.93      0.93     26400
weighted avg       0.93      0.93      0.93     26400

## Author

Klychliiev Kyrylo

[LinkedIn](https://www.linkedin.com/in/kyrylo-klychliiev/)

Ukraine, Chernihiv