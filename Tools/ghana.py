import Tools.preprocessor as pr
import numpy as np 
from natsort import natsorted 
import skimage.exposure
from Unet import Fri_unet
import torch
import itertools
import gdal
from os import listdir
from os.path import isfile, join

def open_ghana_images(path, chanels):
  """
  
  Function opening images of Ghana dowloaded from Google Earth Engine.
  The bands are re-arranged in order to match the one used in data used 
  to create Unet. Could masked NaNs are replaced with 0 for achieve 
  consistenct with model's data.

  Args:
  path: path to folder with Ghana tiff files.
  chanels: number of chanels in data to be returned.
  """
  files = natsorted([f for f in listdir(path) if isfile(join(path, f))])
  ghana_images = []
  for i, j in enumerate(files):
    if j[-4:] == '.tif':
      ghana_images.append(pr.open_tiff_file(path + '/' + str(j)))
  X = []
    #Re-arranging Bands
  if chanels ==3:
    for i in ghana_images:
      input_3_channel = np.flip(i[:, :, 1:4], 2)
      X.append(input_3_channel)
  if chanels ==6:
    for i in ghana_images:
      input_3_channel = np.flip(i[:, :, 1:4], 2)
      nir = np.reshape(i[:, :, 7], (i.shape[0], i.shape[1], 1))
      swir1_2 = i[ :, :, 10:]
      input_6_channel = np.append(np.append(input_3_channel, swir1_2, axis = -1),nir, axis = -1)
      X.append(input_6_channel)
  if chanels ==10:
    for i in ghana_images:
      input_3_channel = np.flip(i[:, :, 1:4], 2)
      nir = np.reshape(i[:, :, 7], (i.shape[0], i.shape[1], 1))
      swir1_2 = i[ :, :, 10:]
      red_edge123 = i[ :, :, 4:7]
      ultra_blue = np.reshape(i[ :, :, 0], (i.shape[0], i.shape[1], 1))
      input_6_channel = np.append(np.append(input_3_channel, swir1_2, axis = -1),nir, axis = -1)
      input_10_channel = np.append(np.append(input_6_channel, red_edge123, axis = -1), ultra_blue, axis = -1)
      X.append(input_10_channel)
    #masking out Nans with 0 for consistency with model data
  for i in range(len(X)):
    if np.isnan(np.sum(X[i])) == True:
      np.nan_to_num(X[i], copy=False, nan=0, posinf=0, neginf=0)
    return X

def get_hist_match_ref(input_path, chanels, image_index):
  """
  Function to get  images used to create Unet to use as 
  a reference to match image histograms of Ghana. 

  Args:
  input_path: path to input images
  chanels: number of channels. image and its histogram matching reference 
           must have equal number of channels.
  image_index: index of an image to which Ghana images will be matched
  """
  if chanels == 3:
    X = pr.open_input_imgs(input_path, 3)
  elif chanels == 6:
    X = pr.open_input_imgs(input_path, 6)
  elif chanels == 10:
    X = pr.open_input_imgs(input_path, 10)
  else:
    print('Images with this number of chanels are not available')
  return X[image_index]

def ghana_analyser(ghana_image, model_images_path, hist_match_index, chanels, model):
  """
  A function taking an original image from ghana transforms it for Unet 
  makes classifications and computes minor ferric iron index.

  The return is histogram matched images of ghana, predictions of 
  pond classifications and array of minor ferric iron index.

  Args:
  ghana_image: Sentinel 2 cloud masked image from Ghana.
  model_images_path: path to images used to build Unet 
  hist_match_index: index of image to match histograms of Ghana image with
  chanels: number of channels in images
  model: pre-loaded Unet model
  """
  preprocess = pr.preprocess
  postprocess = pr.postprocess

  reference = get_hist_match_ref(model_images_path, chanels, hist_match_index)
  matched = skimage.exposure.match_histograms(ghana_image, reference,
                multichannel= True)

  ghana_uniform = pr.split_images_to_uniform_sizes([matched], 64, 64)
  ghana_norm = pr.normalize_01(ghana_uniform)
  ghana_final = np.moveaxis(ghana_norm,  source=-1, destination=1)
  
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  
  preds = np.array([Fri_unet.predict(img, model, preprocess, postprocess, device) for img in ghana_final]) 

  minor_ferric_iron = np.zeros_like(preds).astype(float)

  for i in range(preds.shape[0]):
    minor_ferric_iron[i] = ghana_norm[i, :, :, 0]/ghana_norm[i, :, :, 2]
  
  return ghana_final, preds, minor_ferric_iron


def calculate_area( preds, minor_ferric_iron ):
  """
  Calculates proportion of pond classes and extracts 
  values of minor ferric iron within the ponds. 

  Args:
  preds: numpy array of Unet pond class predictions
  minor_ferric_iron: numpy array of ferric iron index in ponds
  
  """
  active_iron = []
  inactive_iron = []
  transition_iron = []
  for n in range(preds.shape[0]):
    for i in range(preds.shape[1]):
      for j in range(preds.shape[2]):
        if preds[n, i, j] == 1:
          active_iron.append(minor_ferric_iron[n, i, j])
        if preds[n, i, j] == 2:
          transition_iron.append(minor_ferric_iron[n, i, j])
        if preds[n, i, j] == 3:
          inactive_iron.append(minor_ferric_iron[n, i, j])
  
  iron_in_all_ponds = list(itertools.chain(active_iron, transition_iron, inactive_iron))
  active_area = len(active_iron)/len(iron_in_all_ponds)
  inactive_area = len(inactive_iron)/len(iron_in_all_ponds)
  transition_area = len(transition_iron)/len(iron_in_all_ponds)
  
  if len(inactive_iron) != 0:
    return active_area,  transition_area, inactive_area, iron_in_all_ponds
  else:
    return active_area, transition_area,  iron_in_all_ponds

def extract_iron_values_per_category(predictions, ferric_iron_index ):
  """
  Function that splits ferric iron index values per pon class.
  The main use case is testing for differences in distributions.

  Args: 
  predictions: numpy array of Unet pond class predictions
  ferric_iron_index: numpy array of ferric iron index in ponds
  """
  active_ferric_iron = []
  transition_ferric_iron = []
  inactive_ferric_iron = []
  non_pond_ferric_iron = []

  for n in range(predictions.shape[0]):
    for i in range(predictions.shape[1]):
      for j in range(predictions.shape[2]):
        if predictions[n, i, j] == 0:
          non_pond_ferric_iron.append(ferric_iron_index[n, i, j])
        if predictions[n, i, j] == 1:
          active_ferric_iron.append(ferric_iron_index[n, i, j])
        if predictions[n, i, j] == 2:
          transition_ferric_iron.append(ferric_iron_index[n, i, j])
        if predictions[n, i, j] == 3:
          inactive_ferric_iron.append(ferric_iron_index[n, i, j])
  
  if len(inactive_ferric_iron) != 0:
    return active_ferric_iron,  transition_ferric_iron, inactive_ferric_iron, non_pond_ferric_iron
  else:
    return active_ferric_iron, transition_ferric_iron, non_pond_ferric_iron

def open_basemap(path):
  """
  Function that opens basemaps from Planet
  """
  raster = gdal.Open(path)
    
  band = raster.GetRasterBand(1)
  nodata = band.GetNoDataValue()
  rasterArray = raster.ReadAsArray()
  #Create a masked array for making calculations without nodata values
  rasterArray = np.ma.masked_equal(rasterArray, nodata)

  basemap = np.array(rasterArray)
  basemap_final = np.moveaxis(basemap,  source=0, destination=-1)
  #To speed up the processing we select a small region with most dence
  #gold mining
  basemap_final = basemap_final[2500:3300, 1500:2500, :3]

  return basemap_final