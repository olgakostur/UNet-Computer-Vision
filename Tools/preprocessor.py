import numpy as np
from natsort import natsorted 
import gdal
import os
import math
import torch
import cv2
from os import listdir
import scipy.io
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.externals._pilutil import bytescale

def open_tiff_file(path):
  """
  Function opening tiff file with gdal
  and converts it to numpy array.
  """

  ds = gdal.Open(path)
  
  img1 = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount),
                 )
  # Loop over all bands in dataset
  for b in range(ds.RasterCount):
      # GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
      band = ds.GetRasterBand(b + 1)
      # Read in the band's data into the third dimension of our array
      img1[:, :, b] = band.ReadAsArray() 
  return img1

def open_labeled_imgs(path):
  """ 
  Function opening labeled images with single band
  """

  y = []
  labeled_imgs = natsorted([f for f in listdir(path) if isfile(join(path, f))])
  for i, j in enumerate(labeled_imgs):
    y.append(open_tiff_file(path + '/' + str(j)))
  
  return y

def open_input_imgs(path, channels):

  """
  Function opens all input tiff files and creates a 3D array of them. 
  Nesessary for folder structure in which images were organised.
  """

  subfolders = natsorted([ f.path for f in os.scandir(path) if f.is_dir() ])
  X = []
  for i, k in enumerate(subfolders):
    if channels == 3:
      input_files = [f for f in listdir(subfolders[i]) if isfile(join(subfolders[i], f))]
      X.extend(open_tiff_file(k + '/' + x)  for x in input_files if x[-3:] == 'tif')
    elif channels == 6:
      input_files = [f for f in listdir(subfolders[i]) if isfile(join(subfolders[i], f))]
      X.extend(scipy.io.loadmat(k + '/' + x)['new']  for x in input_files if x[-7:] == 'ch6.mat')
    elif channels == 10:
      input_files = [f for f in listdir(subfolders[i]) if isfile(join(subfolders[i], f))]
      X.extend(scipy.io.loadmat(k + '/' + x)['new']  for x in input_files if x[-8:] == 'ch10.mat')
    else:
      print('images with this number of channels is not available')
  return X

def split_images_to_uniform_sizes(images,  min_width, min_height):
  """
  Function cuts images to equal sizes for simpler and quicker processing 
  by model.

  Args:
  images: list if arrays of varying sizes.
  min_width, min_heights: intergers of size of images to cut to
  """
  #number of channels will be the same in all images, 
  #so any could be chosen to compute this
  channels = images[0].shape[2]

  padded = []
  for i in images:
    #if an image is more than 20% off to xn in size, then it
    #then it will be cut to nearest value allowing to split image in
    #several sections of shape min_width x min_height
    if math.modf(i.shape[0]/min_height)[0] <0.8 and math.modf(i.shape[1]/min_width)[0] <0.8:
      padded.append(i[:min_height * int(math.modf(i.shape[0]/min_height)[1]), :min_width * int(math.modf(i.shape[1]/min_width)[1]), :])
    #if the image is less than 20% off to xn in size, then
    #it will be padded to not loose valuable data
    if math.modf(i.shape[0]/min_height)[0] >= 0.8 and math.modf(i.shape[1]/min_width)[0] <0.8:
      pad_height =  (math.modf(i.shape[0]/min_height)[1] +1) * min_height - i.shape[0] 
      j = np.pad(i, [(0, int(pad_height)), (0, 0), (0, 0)], mode='constant', constant_values=0)
      j = j[:, :min_width * int(math.modf(i.shape[1]/min_width)[1])]
      padded.append(j)
  #Splittin an padded images into multiple sub-arrays horizontally (column-wise)
  h_sub_arrays = []
  s = [np.hsplit(ex, ex.shape[1]/min_width) for ex in padded] 
  for i in range(len(s)):
    h_sub_arrays.extend(s[i])
  #Split an sub-images into multiple sub-arrays vertically (row-wise).
  v_sub_arrays = []
  s = [np.vsplit(ex, ex.shape[0]/min_height) for ex in h_sub_arrays] 
  for i in range(len(s)):
    v_sub_arrays.extend(s[i])
  #cheking that all images have correct dimentions
  for i in v_sub_arrays:
    if i.shape != (min_height, min_width, channels):
      print('Splitting did not go correctly')
  
  dataset = np.array(v_sub_arrays)
  #we need to remove the channel dimention 
  #in case it is 1, for tensor data construction
  if channels == 1:
    dataset = np.reshape(dataset, (len(v_sub_arrays),min_height, min_width))
  
  return dataset

def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (no clipping)"""
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out

def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out 

def preprocess(img: np.ndarray):
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img

def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    #img = re_normalize(img)  # scale it to the range [0-255]
    return img

def create_model_dataset(label_path, input_path, input_chanels, num_classes):

  """
  Function combining all sub-functions to created the model dataset,
  including all preprocessing steps. 

  Args:
  label path: path to labeled images
  imput_path: path to input images
  input_chanels: interger, number of input chanels
  num_classes: interger, number of input classed 
  """
  
  y = open_labeled_imgs(label_path)
  Y = split_images_to_uniform_sizes(y, 64, 64)
  #removing data with mostly background class = 0
  indices = []
  for i in range(Y.shape[0]):
    if np.count_nonzero(Y[i] == 0)/(64*64) >= 0.9:
      indices.append(i)
  Y_cleaned = np.delete(Y, indices, axis=0)

  if input_chanels == 3:
    X = open_input_imgs(input_path, 3)
  elif input_chanels == 6:
    X = open_input_imgs(input_path, 6)
  elif input_chanels == 10:
    X = open_input_imgs(input_path, 10)
  else:
    print('Images with this number of chanels are not available')
  X = split_images_to_uniform_sizes(X, 64, 64)
  X = normalize_01(X)
  X = np.moveaxis(X,  source=-1, destination=1)
  #removing data with mostly background class = 0
  X_cleaned = np.delete(X, indices, axis=0)
  if num_classes == 4:
    return Y_cleaned, X_cleaned
  #creating dataset with 3 classes by combining 
  # transition and inactive class.
  if num_classes ==3:
    Y_cleaned[Y_cleaned == 3] = 2
    return Y_cleaned,  X_cleaned
  if num_classes == 2:
    Y_cleaned[Y_cleaned == 3] = 1
    Y_cleaned[Y_cleaned == 2] = 1
    return Y_cleaned,  X_cleaned
  else:
    print('No data with this many classes')



def enhance_image_contrast(img, red, blue):
    """
    The function enhances image contrast in RGB, for better visualisation.
    Function most used for plotting and initial exploration of data.

    Args:
    img: image to enhance in numpy array format.
    in multiband images there red, green and blue bands 
    need to be selected to enhace the contrast.
    red: index of red band
    blue: index of blue band
    """
    #get min and max values of rgb bands
    max_val = np.nanmax(img[ :, :, red:blue])
    min_val = np.nanmin(img[ :, :, red:blue])
    
    # Enforce maximum and minimum values
    for inx in range(3, 0, -1):
        img[img[ :, :, inx] > max_val] = max_val
        img[img[ :, :, inx] < min_val] = min_val

    for b in range(3, 0, -1):
        img[ :, :, b] = img[:, :, b] * 1 / (max_val - min_val)
    return img


