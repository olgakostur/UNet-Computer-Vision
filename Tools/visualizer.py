import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
def visualise_results(inputs,  predictions, classes, nth_set, targets = None):
  """ 
  The function visualises the results of UNet predictions 
  
  Args:
  inputs: the Sentinel 2 inputs to the model to predict from
  tragets: the labels to compare predictions with
  predictions: the output of the model 
  classes: the number fo classes wich model is identifying 3 or 4
  nth_set: this parameter eases the scrolling through images by 
           taking the next 6th with each increase by 1
  """
  if classes == 2:
    cmap = colors.ListedColormap(['white', 'red', ])
    labels = [ 'not pond', 'pond']
  if classes == 3:
    cmap = colors.ListedColormap(['white', 'red', 'green'])
    labels = [ 'background', 'active',  'inactive']
  if classes == 4:
    cmap = colors.ListedColormap(['white', 'red', 'blue', 'green'])
    labels = [ '', 'background', 'active', 'transition', 'inactive']
  #re-arranging axis for plotting
  inputs_test_= np.moveaxis(inputs,  source=1, destination=-1)
  fig, axes = plt.subplots(2,3, figsize=[8, 6])
  fig.suptitle('Inputs', fontsize=16)
  for i, ax in enumerate(axes.ravel()):
      im = ax.imshow(inputs_test_[i+nth_set*6][:, :, :3])
      ax.axis('off')
  plt.subplots_adjust(wspace=0.05, hspace=0)
  
  if targets is not None:
    fig, axes = plt.subplots(2,3, figsize=[8,6])
    fig.suptitle('Targets', fontsize=16)
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(targets[i+nth_set*6], cmap = cmap)
        ax.tick_params(bottom=False, left = False, labelbottom=False, labelleft = False)
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax, fraction=0.046, pad=0.04) 
    cbar.ax.locator_params(nbins=classes)
    cbar.ax.set_yticklabels(labels) 
  plt.subplots_adjust(wspace=0.05, hspace=0) 


  fig, axes = plt.subplots(2,3, figsize=[8,6])
  fig.suptitle('Predictions', fontsize=16)
  for i, ax in enumerate(axes.ravel()):
      im = ax.imshow(predictions[i+nth_set*6], cmap = cmap)
      ax.tick_params(bottom=False, left = False, labelbottom=False, labelleft = False)
  cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])

  cbar = fig.colorbar(im, cax=cb_ax) 
  cbar.ax.locator_params(nbins=classes)
  cbar.ax.set_yticklabels(labels) 
  plt.subplots_adjust(wspace=0.05, hspace=0)