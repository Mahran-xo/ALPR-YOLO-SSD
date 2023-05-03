import os
import glob
import cv2 
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fastapi import FastAPI , File , UploadFile
import asyncio
import uvicorn
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import get_file
from PIL import Image
import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util



CUSTOM_MODEL_NAME = 'alpr_ssd_mobnet' 
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('codes', 'workspace'),
    'SCRIPTS_PATH': os.path.join('codes','scripts'),
    'APIMODEL_PATH': os.path.join('codes','models'),
    'ANNOTATION_PATH': os.path.join('codes', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('codes', 'workspace','images'),
    'MODEL_PATH': os.path.join('codes', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('codes', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('codes', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('codes', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('codes', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('codes', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('codes','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('codes', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', '*.jpg')

def iso_plate(image_np):
    # for i in tqdm(range(len(images_path))):
    # img = cv2.imread(images_path)
    # image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = np.array(image_np).copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.1,
                agnostic_mode=False)

    img = Image.fromarray(image_np_with_detections[...,::-1])
    detection_threshold = 0.2
    image = image_np_with_detections
    width = image.shape[1]
    height = image.shape[0]
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    max_score_box = boxes[scores.index(max(scores))]   
    roi = max_score_box*[height, width,height,width]
    region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
    plate = Image.fromarray(region[...,::-1])
    # path = os.path.join(paths['IMAGE_PATH'], 'plates','IMG{}.jpg'.format(i))
    # plate.save(path)
    # img.save(paths['IMAGE_PATH']+'\detected\d'+os.path.basename(images_path[i]))
    return plate


app = FastAPI()

@app.post('/')
async def scoring_endpoint(data:UploadFile=File(...)):
    image_bytes = await data.read()
    image = tf.io.decode_image(image_bytes)
    yhat = iso_plate(image)
    yhat = np.array(yhat)
    return {"plate":json.dumps(yhat.tolist())}