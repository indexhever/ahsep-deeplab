import os
#import tarfile
#import tempfile
#from six.moves import urllib
import argparse
parser = argparse.ArgumentParser()
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.python.platform import gfile

parser.add_argument("--model_folder", help="Model name used")
args = parser.parse_args()

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, pb_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    '''
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break
    tar_file.close()
    '''

    # Extract frozen grapf from pb archive.
    with tf.Session() as sess:
	    model_filename = pb_path
	    with gfile.FastGFile(model_filename, 'rb') as f:
	        graph_def = tf.GraphDef()
	        graph_def.ParseFromString(f.read())    

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    #startTimeRun = time.time()

    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

    #endTimeRun = time.time()

    #print "Time run: " + str((endTimeRun - startTimeRun))

    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def ConfMatrix(resultImg, maskImg, totalArm, totalNotArm, predArmAndIsArm, predArmAndIsNotArm, predNotArmIsArm, predNotArmIsNotArm, totalPixels, backColorSegMap=0, frontColorLabel=1):
  height, width = maskImg.shape
  for y in range(0,height): 
    for x in range(0,width): 
      #print resultImg[y,x]
      #backgroundColor = 14
      #backgroundColor = 30
      if maskImg[y,x] == frontColorLabel:
        totalArm = totalArm + 1
      else:
        totalNotArm = totalNotArm + 1
      #print "color: "
      #print resultImg[y,x]
      # when resultImage value is arm and maskImage value is arm
      if resultImg[y,x] != backColorSegMap and maskImg[y,x] == frontColorLabel:
        predArmAndIsArm = predArmAndIsArm + 1
      
      # when resultImage value is arm and maskImage value is not arm
      if resultImg[y,x] != backColorSegMap and maskImg[y,x] != frontColorLabel:
        predArmAndIsNotArm = predArmAndIsNotArm + 1

      # when resultImage value is not arm and maskImage value is arm
      if resultImg[y,x] == backColorSegMap and maskImg[y,x] == frontColorLabel:
        predNotArmIsArm = predNotArmIsArm + 1

      # when resultImage value is not arm and maskImage value is not arm
      if resultImg[y,x] == backColorSegMap and maskImg[y,x] != frontColorLabel:
        predNotArmIsNotArm = predNotArmIsNotArm + 1

      totalPixels = totalPixels + 1
  return (totalArm, totalNotArm, predArmAndIsArm, predArmAndIsNotArm, predNotArmIsArm, predNotArmIsNotArm, totalPixels)

'''
def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
'''

val = np.loadtxt('/home/conteinerFiles/skin-images/val', dtype=str)

if args.model_folder:
  modelFolder = args.model_folder
else:
  modelFolder = "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_21classes_14batch_mobilenet"

print "Using model: " + modelFolder

pb_path = "/home/conteinerFiles/deeplab/models/research/deeplab/datasets/hsarah/exp/" + modelFolder + "/export/frozen_inference_graph.pb"
logFile = open("/home/conteinerFiles/deeplab/models/research/deeplab/datasets/hsarah/exp/" + modelFolder + '_acc.log', 'a+')
deepModel = DeepLabModel(pb_path)

currentAcc = 0.0
for idx in val:
  imagem = Image.open("/home/conteinerFiles/skin-images/skin-images/" + idx + ".jpg")
  label = Image.open("/home/conteinerFiles/skin-images/masks/" + idx + ".pbm")
  #labelArray = label.load()
  #startTimeOneVis = time.time()
  startTimeRun = time.time()
  resized_image, seg_map = deepModel.run(imagem)
  endTimeRun = time.time()
  print "Time run: " + str((endTimeRun - startTimeRun))
  labelArray = np.array(label)
  #print "label array before resize: "
  #print labelArray.shape
  labelArray = np.resize(labelArray, seg_map.shape)
  labelArray = labelArray * 1
  '''
  for y in range(seg_map.shape[0]):
    for x in range(seg_map.shape[1]):
      if seg_map[y,x] != 0 and seg_map[y,x] != 1:
        print seg_map[y,x]
  
  print "label size: "
  print label.size
  print "image size: "
  print imagem.size
  print "labelArray shape: "
  print labelArray.shape
  print "seg_map shape: "
  print seg_map.shape
  '''
  #endTimeOneVis = time.time()
  seg_map = seg_map.flatten()
  seg_map = np.reshape(seg_map, (seg_map.shape[0], 1))
  labelArray = labelArray.flatten()
  labelArray = np.reshape(labelArray, (labelArray.shape[0], 1))
  #print seg_map.shape
  #print labelArray.shape
  newAcc = float((np.dot(labelArray.T,seg_map) + np.dot(1-labelArray.T,1-seg_map))/float(seg_map.size)*100)
  print idx + " acc: " + str(newAcc) + '%'
  logFile.write(idx + " acc: " + str(newAcc) + '%' + '\n')
  currentAcc = currentAcc + newAcc

totalAcc = float(currentAcc/float(val.size))
print "Total acc: " + str(totalAcc) + '%'
logFile.write("Total acc: " + str(totalAcc) + '%' + '\n')
