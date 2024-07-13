import numpy as np
import os
from skimage.io import imread
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Functions for preparation of dataset

BATCH_SIZE = 64
IMG_SCALING = (3, 3)

# Decode RLE encoded masks
def decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert to 0-based indexing
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Align to RLE direction

# Combine individual masks into a single mask
def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= decode(mask)  # Combine masks
    return all_masks

# Create batches of images and masks
def make_image_gen(in_df, image_path_train, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb, out_mask = [], []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(image_path_train, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb.append(c_img)
            out_mask.append(c_mask)
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0).astype(np.float32)
                out_rgb, out_mask = [], []  # Reset batches

# AUGMENTATION

# Arguments for the ImageDataGenerator for data augmentation
dg_args = dict(
    featurewise_center=False, 
    samplewise_center=False,
    rotation_range=45, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    shear_range=0.01,
    zoom_range=[0.9, 1.25],  
    horizontal_flip=True, 
    vertical_flip=True,
    fill_mode='reflect',
    data_format='channels_last'
)

# Initialize image and label data generators with the defined arguments
image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)
# Function to create an augmented data generator
def create_aug_gen(in_gen):
    for in_x, in_y in in_gen:
        # Synchronize the seeds to ensure the same augmentation for images and labels
        seed = np.random.choice(range(10000))  
        
        # Generate augmented images
        g_x = image_gen.flow(
            in_x, 
            batch_size=in_x.shape[0], 
            seed=seed, 
            shuffle=True
        )
        
        # Generate augmented labels
        g_y = label_gen.flow(
            in_y, 
            batch_size=in_x.shape[0], 
            seed=seed, 
            shuffle=True
        )

        # Yield the next batch of augmented images and labels
        yield next(g_x), next(g_y)


# Constants for the Focal Loss
ALPHA = 0.8
GAMMA = 2

# Function to compute the Focal Loss
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    # Flatten the inputs and targets to 1D
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    # Compute the Binary Cross-Entropy loss
    BCE = K.binary_crossentropy(targets, inputs)
    # Compute the exponent of the negative BCE
    BCE_EXP = K.exp(-BCE)
    # Compute the Focal Loss
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)
    
    return focal_loss

# Dice score function 

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(test_dir, img)
    img = cv2.imread(rgb_path)
    img = img[::IMG_SCALING[0], ::IMG_SCALING[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred