import tensorflow as tf
import logging
import argparse
from easydict import EasyDict as edict
import json

print("TensorFlow version:", tf.__version__)
from tensorflow.keras import optimizers, losses

from pathlib import Path
import sys
import os

from model.mvcnn import mvcnn
from model.resnet import resnet
from model.efficientnet import efficientnet
from model.mobilenet import mobilenet
from config import cfg
from data_loader import load_dir_test_data as load_data

tf.get_logger().setLevel(logging.ERROR)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

RESNET = "resnet"


logger = logging.getLogger(__name__)

# Cost function
loss_object = losses.SparseCategoricalCrossentropy()

# Optimizer
optimizer = optimizers.Adam(learning_rate=0.0001)


def eval(hyp):
    """Evaluate multiview model predictions. Test data class has to be the same as training.

    Args:
        hyp (dict): A dictionary of hyperparameters required for evaluation.
    """
    # Data Loader
    test_ds, class_names = load_data(
        hyp.test_dir, hyp.label_mode, int(hyp.batch_size)
    )
    # Uncomment to check sample
    # for mvs, label in test_ds:
    #     print("Sample mvs: ")
    #     print(mvs)
    #     print("Sample label: ")
    #     print(label)
    #     break
    # return

    # Load / Initialize Model
    print("Class length - ", len(class_names), class_names)
    # Compiling issues - https://stackoverflow.com/questions/74667876/typeerror-weight-decay-is-not-a-valid-argument-kwargs-should-be-empty-for-opt
    if hyp.model == 'resnet':
        model = resnet(num_classes=len(class_names))
    elif hyp.model == 'efficientnet':
        model = efficientnet(num_classes=len(class_names))
    elif hyp.model == 'mobilenet':
        model = mobilenet(num_classes=len(class_names))
    elif hyp.model == 'mvcnn':
        model = mvcnn( num_classes=len(class_names))

    input_shape = (None, int(hyp.views), int(hyp.img_size), int(hyp.img_size), 3)
    model.build(input_shape=input_shape)
    model.compile(
        # optimizer=optimizer,
        loss=loss_object,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary(expand_nested=True)
    model.load_weights(hyp.model_weights)
    metrics = model.evaluate(test_ds,
                             return_dict=True)
    print(metrics)
    with open(hyp.metrics_path, 'w') as f:
        json.dump(metrics,f)
    print("Evaluation metric stored in - ", hyp.metrics_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=False, default= "../../../../data/test", help="")
    parser.add_argument("--model_weights", type=str, required=False, default= "../../../../data/model/weights", help="")
    parser.add_argument("--metrics_path", type=str, required=False, default= "../../../../data/model/metrics.json", help="")
    parser.add_argument("--views", type=str, required=False, default=6, help="")
    parser.add_argument("--model", type=str, required=False, default= "resnet", help="")
    parser.add_argument("--batch_size", type=str, required=False, default= 1, help="")
    parser.add_argument("--random_state", type=int, required=False, default=42, help="")
    parser.add_argument("--img_size", type=int, required=False, default=256)
    parser.add_argument("--label_mode", type=str, required=False, default="categorical", help="")
    
    opt = parser.parse_args()
    argparse_dict = vars(opt)
    
    cfg = edict(argparse_dict)
    
    eval(cfg)
