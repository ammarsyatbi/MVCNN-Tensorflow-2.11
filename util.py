from fileinput import filename
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import json
import re
import os
from os.path import join as osp

from config import cfg


URL_RE = r"https?:\/\/(www\.)?([-a-zA-Z0-9@:%._\+~#=]{1,256})\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#&//=]*)"


def decode_img(image_bytes, invert=True):
    buf = np.ndarray(shape=(1, len(image_bytes)), dtype=np.uint8, buffer=image_bytes)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if invert:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_class_names():
    with open("./src/config.json") as f:
        conf = json.loads(f.read())
        classes = conf["class_names"]

    return classes


def decode_prediction(prediction, class_names):
    pred_index = np.argmax(prediction)
    name = class_names[pred_index]
    name = " ".join([n.capitalize() for n in name.split("_")])
    return name


def imgs_to_mvs(imgs):
    # mv = mv.numpy()
    views = []
    for img in imgs:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = np.array(img) / 255.
            img = tf.image.resize(img, [int(cfg.img_size), int(cfg.img_size)])
            img = tf.image.convert_image_dtype(
                img,
                tf.float32,
            )
        except Exception as e:
            print(e)
            img = np.zeros((int(cfg.img_size), int(cfg.img_size), 3), np.float32)
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        views.append(img)

    return tf.data.Dataset.from_tensor_slices([views])


def dir_to_mvs(directory):
    views = os.listdir(directory)
    paths = [osp(directory, v) for v in views if v != ".DS_Store"]

    assert (
        len(paths) == cfg.views
    ), f"List of filenames should match number of views {cfg.views}"

    imgs = get_multiviews(paths)
    return tf.data.Dataset.from_tensor_slices([imgs])


def get_bucket_key(img_url):
    found = re.search(URL_RE, img_url)
    if found:
        domain, key = found.group(2), found.group(3)
        bucket = domain.split(".")[0]

    return bucket, key


def get_multiviews(mv=None):
    # mv = mv.numpy()
    views = []
    for v in mv:
        try:
            if type(v) != str:
                v = v.decode()

            img = cv2.imread(v)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = np.array(img) / 255.
            img = tf.image.resize(img, [int(cfg.img_size), int(cfg.img_size)])
            img = tf.image.convert_image_dtype(
                img,
                tf.float32,
            )
        except Exception as e:
            print(f"Corrupted image - {v}")
            print(f"Filename type - {type(v)}")
            print(e)
            img = np.zeros((int(cfg.img_size), int(cfg.img_size), 3), np.float32)
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        views.append(img)

    return views


def read_mutliviews(mv):
    return tf.py_function(get_multiviews, [mv], tf.float32)


def labels_to_dataset(labels, num_classes, label_mode="categorical"):
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    if label_mode == "binary":
        label_ds = label_ds.map(
            lambda x: array_ops.expand_dims(math_ops.cast(x, "float32"), axis=-1)
        )
    ## Uncomment if want to have one-hot label
    # elif label_mode == "categorical":
    # label_ds = label_ds.map(lambda x: array_ops.one_hot(x, num_classes))

    return label_ds


def directory_to_multiviews(directory):
    multi_views = []
    classes = [c for c in sorted(os.listdir(directory)) if c != ".DS_Store"]
    # print(f"Multiviews classes - {classes}")
    for subdir in classes:
        # print(f"Multiviews subdir - {subdir}")
        mv_paths = [
            osp(directory, subdir, mv_path)
            for mv_path in os.listdir(osp(directory, subdir))
            if mv_path != ".DS_Store"
        ]
        for mv_path in mv_paths:
            mv = [osp(mv_path, v) for v in os.listdir(mv_path)]
            multi_views.append(mv)
    return np.array(multi_views)


def directory_to_multiviews(directory):
    multi_views = []
    classes = [c for c in sorted(os.listdir(directory)) if c != ".DS_Store"]
    # print(f"Multiviews classes - {classes}")
    for subdir in classes:
        # print(f"Multiviews subdir - {subdir}")
        mv_paths = [
            osp(directory, subdir, mv_path)
            for mv_path in os.listdir(osp(directory, subdir))
            if mv_path != ".DS_Store"
        ]
        for mv_path in mv_paths:
            mv = [osp(mv_path, v) for v in os.listdir(mv_path)]
            multi_views.append(mv)
    return np.array(multi_views)


def img_generator(mvs):
    for mv in mvs:
        imgs = get_multiviews(mv)
        yield [imgs]


def label_generator(labels):
    for label in labels:
        yield [label]


def generate_dataset(mvs, labels, num_classes, label_mode, batch_size):
    mvs, labels = np.array(mvs), np.array(labels, dtype=np.int64)

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


def directory_to_labels(directory):
    class_names = [
        subdir
        for subdir in sorted(os.listdir(directory))
        if os.path.isdir(osp(directory, subdir)) and subdir != ".DS_Store"
    ]
    class_names = sorted(class_names)

    labels = []
    for subdir in class_names:
        # multi view directory
        for mv_dir in os.listdir(osp(directory, subdir)):
            if os.path.isdir(osp(directory, subdir, mv_dir)) and mv_dir != ".DS_Store":
                labels.append(subdir)

    labels = [int(class_names.index(label)) for label in labels]
    labels = np.array(labels)
    return labels, class_names


def generate_csv_dataset(mvs, labels):
    mvs, labels = np.array(mvs), np.array(labels, dtype=np.int8)

    img_dataset = tf.data.Dataset.from_tensor_slices(mvs).map(
        lambda x: read_s3_mutliviews(x)
    )

    # Labels already in one-hot
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((img_dataset, labels_dataset))
    return dataset


def load_train_data(
    directory, val_split, label_mode="categorical", batch_size=1, random_state=42
):
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


def load_test_data(
    directory,
    label_mode="categorical",
    batch_size=1,
):
    labels, class_names = directory_to_labels(directory)
    mvs = directory_to_multiviews(directory)

    print(f"SAMPLE LENGTH - {len(mvs)}")
    print(f"LABEL LENGTH - {len(labels)}")
    print(f"LABEL CLASSNAMES - {class_names}")

    # with open("../src/config.json") as f:
    #     classes = json.loads(f.read())

    classes = cfg.class_names

    for c in class_names:
        if c not in classes:
            raise KeyError("Class names not available in training")

    num_classes = int(len(class_names))

    test_ds = generate_dataset(mvs, labels, num_classes, label_mode, batch_size)

    return test_ds, class_names


def dataframe_to_dataset(df):
    class_names = sorted(df.brand_model.drop_duplicates().tolist())
    df.labels = df.brand_model.apply(lambda x: class_names.index(x))

    labels = to_categorical(df.labels.tolist())

    mvs = df.views.str.split(",").tolist()

    return mvs, labels, class_names
