# Winstars_Technology_Test_Task
## Description
This basic purpose of this project is to make a model that will make a segmentation of an input image. To accomplish this U-Net model was trained on a kaggle competitions Airbus Ship Detection Challenge Dataset. 

'data_analysis.jpynb' - jupyter notebook with exploratory data analysis of the dataset;

'model.py' - file with model architecture;

'utils.py' - file with useful functions;

'train.py' - file for training model;

'test.py' - file for testing pretrained model.

## To run task
  1. Save this repo
  2. Install all requirements
  3. In the 'data_analisys.jpynb' replace path to your dataset path in all cells.
  4. Run 'data_analisys.jpynb' file.
### To train model
  1. Open file 'train.py'
  2. In the fifth row replace path to your dataset path (format of paths have to be path/to/images/* and path/to/masks/*)
  3. Run the file
### To test model
  1. Download pretrained weights unet.h5 from [Google Drive](https://drive.google.com/file/d/149A1EH9innjjH4TVJ6P4vlgyVQjtbEKK/view?usp=sharing).
  2. Open file 'test.py'
  3. Replase paths in rows 9, 14, 15, 18
  4. Run code
