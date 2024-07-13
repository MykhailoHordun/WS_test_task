# Winstars_Technology_Test_Task
## Description
This basic purpose of this project is to make a model that will make a segmentation of an input image. To accomplish this U-Net model was trained on a kaggle Airbus Ship Detection Challenge Dataset. 

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
  2. Replace all paths
  3. Run the file
### To test model
  1. Download pretrained weights unet.h5 from [Google Drive](https://drive.google.com/drive/folders/1e8I07_UsbpmXiO8Nm-iI3BCI-_rgaEjR?usp=sharing).
  2. Open file 'test.py'
  3. Replace all paths
  4. Run code
