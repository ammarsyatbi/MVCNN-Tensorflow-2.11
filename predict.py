from util import dir_to_mvs, load_class_names, decode_prediction

from model.mvcnn import mvcnn
from model.resnet import resnet
from model.efficientnet import efficientnet
from model.mobilenet import mobilenet

from config import cfg

import tensorflow as tf
import logging
import argparse
from easydict import EasyDict as edict

tf.get_logger().setLevel(logging.ERROR)

class_names = load_class_names()

# Load / Initialize Model
model = mobilenet(num_classes=len(class_names))
input_shape = (None, int(cfg.views), int(cfg.img_size), int(cfg.img_size), 3)
model.build(input_shape=input_shape)
# model.compile()
# model.summary(expand_nested=True)
model.load_weights(cfg.model_weights)


def predict(hyp):
    """Run model classification on 6 multiviews 

    Args:
        hyp (dict): Parameters for prediction

    Returns:
        str: decoded label in string
    """
    
    mvs = dir_to_mvs(hyp.mvs)
    
    pred = model.predict(mvs)
    class_name = decode_prediction(pred, class_names)
    print(class_name)
    return class_name
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvs", type=str, required=False, default= "../data/mvs", help="")
    parser.add_argument("--model_weights", type=str, required=False, default= "../data/model/weights", help="")
    parser.add_argument("--model", type=str, required=False, default= "mobilenet", help="")
    parser.add_argument("--views", type=str, required=False, default=6, help="")
    parser.add_argument("--img_size", type=int, required=False, default=224)
    
    opt = parser.parse_args()
    argparse_dict = vars(opt)
    
    cfg = edict(argparse_dict)
    
    predict(cfg)