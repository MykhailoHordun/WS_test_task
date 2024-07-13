from model import *
from utils import *
import matplotlib.pyplot as plt

test_data_path = "D:/TestData/test" # Replace with the path to the test images dir
weights = './model_weights.best.weights.h5' # Replace with the path to the pretrained U-net model weights
model = unet(pretrained_weights=weights)

test_imgs = ['00dc34840.jpg', '00c3db267.jpg', '00aa79c47.jpg', '0ad121a1c.jpg']

# Iterate over the images and pridictiong masks 

rows = 1
columns = 2
for i in range(len(test_imgs)):
    img, pred = gen_pred(test_data_path, test_imgs[i], model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
    plt.show()
