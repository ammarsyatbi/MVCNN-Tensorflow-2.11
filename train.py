import tensorflow as tf
import logging
import argparse
from easydict import EasyDict as edict
import pickle

print("TensorFlow version:", tf.__version__)
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from pathlib import Path
import sys
import os

from model.mvcnn import mvcnn
from model.resnet import resnet
from model.efficientnet import efficientnet
from model.mobilenet import mobilenet

from config import cfg
from data_loader import load_dir_train_data as load_data


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

RESNET = "resnet"


logger = logging.getLogger(__name__)


def train(hyp):
    """Train multiview image classification model

    Args:
        hyp (dict): A dictionary of hyperparameters required for training process.
    """
    # Data Loader
    train_ds, val_ds, class_names = load_data(
        hyp.train_dir, float(hyp.val_split), hyp.label_mode, int(hyp.random_state), int(hyp.batch_size)
    )
    
    # Uncomment to check sample
    # for inputs, label in train_ds:
    #     print("Sample label: ")
    #     print(label)
    #     break
    # return
    
    # Load / Initialize Model
    print("Class length - ", len(class_names), class_names)

    # Cost function
    loss_object = losses.SparseCategoricalCrossentropy()

    # Optimizer
    optimizer = optimizers.Adam(learning_rate=float(hyp.learning_rate))

    # callbacks
    callbacks = [
        # TensorBoard(log_dir=hyp.log_dir, histogram_freq=1),
        ModelCheckpoint(filepath=hyp.ckpt_path, monitor='sparse_categorical_accuracy', mode='max', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1),
        # ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1),
    ]

    if hyp.model == 'resnet':
        model = resnet(pretrained=hyp.pretrained, num_classes=len(class_names))
    elif hyp.model == 'efficientnet':
        model = efficientnet(pretrained=hyp.pretrained, num_classes=len(class_names))
    elif hyp.model == 'mobilenet':
        model = mobilenet(pretrained=hyp.pretrained, num_classes=len(class_names))
    elif hyp.model == 'mvcnn':
        model = mvcnn(pretrained=hyp.pretrained, num_classes=len(class_names))

    input_shape = (None, int(hyp.views), int(hyp.img_size), int(hyp.img_size), 3)
    model.build(input_shape=input_shape)
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary(expand_nested=True)
    
    history = model.fit(train_ds, 
                        validation_data=val_ds,
                        epochs=int(hyp.epochs),
                        max_queue_size=int(hyp.max_queue_size),
                        # batch_size=int(hyp.batch_size),
                        # validation_batch_size=int(hyp.validation_batch_size),
                        workers=int(hyp.workers),
                        verbose=1, 
                        use_multiprocessing=bool(hyp.use_multiprocessing),
                        callbacks=callbacks)
    model.save_weights(hyp.model_weights)
    
    with open(os.path.join(hyp.history_path), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    print("NUMBER OF GPUs - ", len(physical_devices))
    # Uncomment to disable memory preload
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=False, default= "../data/train", help="")
    parser.add_argument("--model_weights", type=str, required=False, default= "../data/model/weights", help="")
    parser.add_argument("--log_dir", type=str, required=False, default="../data/model/logs", help="")
    parser.add_argument("--ckpt_path", type=str, required=False, default="../data/model/ckpt/ckpt", help="")
    parser.add_argument("--history_path", type=str, required=False, default="../data/model/history.pkl", help="")
    parser.add_argument("--learning_rate", type=float, required=False, default= 0.0001, help="")
    parser.add_argument("--views", type=str, required=False, default=6, help="")
    parser.add_argument("--model", type=str, required=False, default= "resnet", help="")
    parser.add_argument("--epochs", type=str, required=False, default= 1, help="")
    parser.add_argument("--batch_size", type=str, required=False, default= 1, help="")
    parser.add_argument("--validation_batch_size", type=str, required=False, default= 1, help="")
    parser.add_argument("--verbose", type=int, required=False, default= 10, help="")
    parser.add_argument("--resume", type=str, required=False, default= "", help="")
    parser.add_argument("--pretrained", type=str, required=False, default= "", help="")
    parser.add_argument("--val_split", type=float, required=False, default=0.2, help="")
    parser.add_argument("--random_state", type=int, required=False, default=42, help="")
    parser.add_argument("--label_mode", type=str, required=False, default="categorical", help="")
    parser.add_argument("--img_size", type=int, required=False, default=224)
    parser.add_argument("--max_queue_size", type=int, required=False, default=1)
    parser.add_argument("--workers", type=int, required=False, default=1)
    parser.add_argument("--use_multiprocessing", type=bool, required=False, default=1)
    opt = parser.parse_args()
    argparse_dict = vars(opt)
    
    cfg = edict(argparse_dict)
    
    train(cfg)
