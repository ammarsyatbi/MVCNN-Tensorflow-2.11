import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import cfg
import pandas as pd
from util import *


def img_generator(mvs):
    """A generator to retun a list of images in array of float. 

    Args:
        mvs (list): Array of array of images

    Yields:
        list: Multiviews of images 
    """
    for mv in mvs:
        imgs = get_multiviews(mv)
        yield [imgs]


def label_generator(labels):
    """A generator to retun a list of label in scalar value

    Args:
        labels (list): Array of label in scalar vlue

    Yields:
        list: A scalar value wrapped in array 
    """
    for label in labels:
        yield [label]


def generate_dataset(mvs, labels, num_classes, label_mode, batch_size):
    """Transform inputs into tensorflow dataset

    Args:
        mvs (list): Mutliviews - array of array of images
        labels (list): Array of encoded class. A scalar value in integer.
        num_classes (int): Number of unique label.
        label_mode (str): Type of label. Transformation will applied based on type chosen 
        batch_size (int): Size of dataset to process per steps

    Returns:
        tf.dataset: A tensorflow dataset of x and y combined
    """
    mvs, labels = np.array(mvs), np.array(labels, dtype=np.int64)

    # Uncomment if need data to be in one-hot encoded
    # img_dataset = tf.data.Dataset.from_tensor_slices(mvs).map(
    #     lambda x: read_mutliviews(x)
    # )
    # labels_dataset = labels_to_dataset(labels, num_classes, label_mode)

    labels_dataset = tf.data.Dataset.from_generator(
        label_generator,
        args=[labels],
        output_types=(tf.int64),
        output_shapes=(None,),
        name="label_generator",
    )

    img_dataset = tf.data.Dataset.from_generator(
        img_generator,
        args=[mvs],
        output_types=(tf.float32),
        output_shapes=(None, cfg.views, cfg.img_size, cfg.img_size, 3),
        name="image_generator",
    )

    dataset = tf.data.Dataset.zip((img_dataset, labels_dataset)).batch(
        batch_size=batch_size
    )

    return dataset


#### Train Loader ####
def load_dir_train_data(
    directory, val_split, label_mode="categorical", batch_size=1, random_state=42
):
    """Load training data based on directory structure. It has to be in specific structure as:
    |--training_directory
        |--label_0
        |--label_1
        |--label_n
            |--mv_n
                |--image_0.jpg
                |--image_1.jpg
                |--image_n.jpg
                

    Args:
        directory (str): Path where training data located
        val_split (float): percentage of training data to convert to validation dataset
        label_mode (str, optional): Type of label. Transformation will applied based on type chosen . Defaults to "categorical".
        batch_size (int, optional): Size of dataset to process per steps. Defaults to 1.
        random_state (int, optional): A number to record randomization applied to dataset shuffle. Defaults to 42.

    Returns:
        tf.dataset: A tensorflow dataset of x and y combined for both training and validation
    """
    labels, class_names = directory_to_labels(directory)
    mvs = directory_to_multiviews(directory)

    print(f"SAMPLE LENGTH - {len(mvs)}")
    print(f"LABEL LENGTH - {len(labels)}")
    print(f"LABEL CLASSNAMES - {class_names}")

    x_train, x_test, y_train, y_test = train_test_split(
        mvs, labels, test_size=val_split, random_state=random_state
    )

    num_classes = int(len(class_names))

    train_ds = generate_dataset(x_train, y_train, num_classes, label_mode, batch_size)
    val_ds = generate_dataset(x_test, y_test, num_classes, label_mode, batch_size)

    return train_ds, val_ds, class_names


def load_csv__train_data(
    csv_path="../../data/cbm/dataset/csv/train.csv", val_split=0.1, random_state=42
):
    """Load training data from csv

    Args:
        csv_path (str): Path location of csv file located in local. Defaults to "../../data/cbm/dataset/csv/train.csv".
        val_split (float): Percentage of training data to convert to validation dataset. Defaults to 0.1.
        random_state (int): A number to record randomization applied to dataset shuffle. Defaults to 42.

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(csv_path)
    mvs, labels, class_names = dataframe_to_dataset(df)

    x_train, x_test, y_train, y_test = train_test_split(
        mvs, labels, test_size=val_split, random_state=random_state
    )

    print(f"SAMPLE LENGTH - {len(mvs)}")
    print(f"LABEL LENGTH - {len(labels)}")

    train_ds = generate_csv_dataset(x_train, y_train)
    val_ds = generate_csv_dataset(x_test, y_test)

    return train_ds, val_ds, class_names


#### Test Loader ####
def load_dir_test_data(
    directory,
    label_mode="categorical",
    batch_size=1,
):
    """Load test data based on directory structure. It has to be in specific structure as:
    |--test_directory
        |--label_0
        |--label_1
        |--label_n
            |--mv_n
                |--image_0.jpg
                |--image_1.jpg
                |--image_n.jpg

    Args:
        directory (_type_): _description_
        label_mode (str, optional): _description_. Defaults to "categorical".
        batch_size (int, optional): _description_. Defaults to 1.

    Raises:
        KeyError: _description_

    Returns:
        _type_: _description_
    """
    labels, class_names = directory_to_labels(directory)
    mvs = directory_to_multiviews(directory)

    print(f"SAMPLE LENGTH - {len(mvs)}")
    print(f"LABEL LENGTH - {len(labels)}")
    print(f"LABEL CLASSNAMES - {class_names}")

    stored_classes = load_class_names()

    for c in class_names:
        if c not in stored_classes:
            raise KeyError("Class names not available in training")

    num_classes = int(len(class_names))

    test_ds = generate_dataset(mvs, labels, num_classes, label_mode, batch_size)

    return test_ds, class_names
