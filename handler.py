from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import json
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from model.mvcnn import *
from model.resnet import *

import numpy as np
import tensorflow as tf
import cv2

from sagemaker_inference import (
    default_inference_handler
)
import time
import os
from sagemaker_inference import logging
from config import cfg

logger = logging.get_logger()

class MRCNNHandler(default_inference_handler.DefaultInferenceHandler):
    def __init__(self):
        self.MODEL_DIR = "model/mvcnn"
        self.CONTENT_TYPE = None
        self.DEVICE = "/cpu:0"
        
    def default_model_fn(self, model_dir):
        logger.info("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        logger.info(f"Loading model from {model_dir}")
        logger.info(f"Containing files - {os.listdir(model_dir)}")
        
        dir_log = ""
        for x in os.walk(model_dir):
            dir_log = dir_log + f"{x[0]} - {x[2]} \n"
        logger.info(dir_log)
        
        # physical_devices = tf.config.list_physical_devices("GPU")
        # for gpu in physical_devices:
        #     tf.config.set_memory_growth(gpu, True)
        
        model = resnet(pretrained=cfg.pretrained, num_classes=len(cfg.class_names))
        input_shape = (None, int(cfg.views), int(cfg.img_size), int(cfg.img_size), 3)
        model.build(input_shape=input_shape)
        model.summary(expand_nested=True)
        model.load_model(self.MODEL_DIR)
        
        return model
    
    def default_input_fn(self, input_data, content_type):
        logger.info(f"Input content type - {content_type} , {type(input_data)}")
        self.CONTENT_TYPE = content_type #application/x-image
        img_bytes = bytes(input_data)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        logger.info(f"Image Input decoded, height - {height} , width - {width}")
        
        return image
    
    def default_predict_fn(self, data, model):
        image = data
        t1 = time.time()
        #TODO: add return bounding box
        pred = model.predict(image)
        t2 = time.time()
        logger.info(f"Time taken for prediction {t2-t1:2f} seconds")
        
        return pred
    
    def default_output_fn(self, prediction, accept):
        img_bytes = cv2.imencode('.jpg', prediction)[1].tobytes()
        logger.info(f"Image output encoded")
        
        return img_bytes, self.CONTENT_TYPE
