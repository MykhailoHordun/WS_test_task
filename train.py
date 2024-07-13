from model import *
from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Dataset loading
data_path = "D:/TestData/train/" # Replace with the path for the training dataset
masks = pd.read_csv("D:/TestData/train_ship_segmentations_v2.csv") # Replace with the path for the masks file
balanced_train_df = pd.read_csv("./balanced_data.csv")

# Dataset preparating

# Train/Val split
train_ids, valid_ids = train_test_split(balanced_train_df, 
                 test_size = 0.05, 
                 stratify = balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids, right_on="ImageId", left_on="ImageId")
valid_df = pd.merge(masks, valid_ids, right_on="ImageId", left_on="ImageId")

valid_gen = make_image_gen(valid_df, data_path)
valid_x, valid_y = next(valid_gen)


model = unet() # creating a U-net model

weight_path = "./model_weights.best.weights.h5"

# Creating the ModelCheckpoint callback with the corrected file path
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             mode='max', save_weights_only=True)

# Creating the ReduceLROnPlateau callback
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, 
                                   patience=3, mode='max', 
                                   min_delta=0.0001, cooldown=2, min_lr=1e-6)

# Creating the EarlyStopping callback
early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=15)

# Combining all callbacks into a list
callbacks_list = [checkpoint, early, reduceLROnPlat]

# training model

def fit():
    model.compile(optimizer=Adam(1e-3, decay=1e-6), loss = FocalLoss, metrics=[dice_coef])
    
    aug_gen = create_aug_gen(make_image_gen(train_df, data_path))
    loss_history = [model.fit(aug_gen,
                                 steps_per_epoch=50,
                                 epochs=100,
                                 validation_data=valid_gen,
                                 callbacks=callbacks_list
                                           )]
    return loss_history

loss_history = fit()