#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture("test.mp4")
out = None




class MotionorNot:
	def __init__(self):
		self.points = []
		self.motionornot = []
		self.motionornot_smooth = []
		
		
	def motion(self, points):
		n_prevpoints = 4
		self.points.append(points)
		
		if len(self.points) > n_prevpoints:
			points_size = len(self.points)
			self.motionornot.append(abs(self.points[points_size-n_prevpoints])-abs(self.points[points_size-1]))	
			
			if len(motionornot) > n_prevpoints:
				for d in range(0, len(motionornot), n_prevpoints):
					self.motionornot_smooth.append(sum(motionornot[d:n_prevpoints+d]))
					
					
	


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



motionornot = MotionorNot()
pd = DataFrame({'mortionornot': [[], []]})

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490;
    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      print(frame)
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter("./media/test_out.avi", 0, 25.0, (w, h))


      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      elapsed_time = time.time() - start_time
      print('inference time cost: {}'.format(elapsed_time))
      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)
      # Visualization of the results of a detection.
      a = vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
          
      motionornot.motion(a)
      out.write(image)


    cap.release()
    out.release()
	pd["motionornot"] = [range(len(motionornot.motionornot_smooth)), motionornot.motionornot_smooth]
	pd.to_pickle("sleep_data.pkl")
	
	
	
