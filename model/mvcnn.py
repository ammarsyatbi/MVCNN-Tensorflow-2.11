import tensorflow as tf

print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    ReLU,
    MaxPool2D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    Dropout,
    InputLayer,
    Rescaling,
    Resizing
)
from tensorflow.keras import Sequential
from tensorflow import math
from config import cfg
class MVCNN(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(MVCNN, self).__init__()
        # Base Model
        self.features = Sequential(name="mvcnn")
        self.features.add(InputLayer(input_shape=((int(cfg.img_size)), int(cfg.img_size),3)))
        self.features.add(Conv2D(64, kernel_size=11, strides=4, padding="same"))
        self.features.add(ReLU())
        self.features.add(MaxPool2D(pool_size=3, strides=2))

        self.features.add(Conv2D(192, kernel_size=5, padding="same"))
        self.features.add(ReLU())
        self.features.add(MaxPool2D(pool_size=3, strides=2))

        self.features.add(Conv2D(384, kernel_size=3, padding="same"))
        self.features.add(ReLU())

        self.features.add(Conv2D(256, kernel_size=3, padding="same"))
        self.features.add(ReLU())

        self.features.add(Conv2D(256, kernel_size=3, padding="same"))
        self.features.add(ReLU())
        self.features.add(MaxPool2D(pool_size=3, strides=2))
        self.features.add(Flatten())

        
        # Classifier
        self.classifier = Sequential()
        self.classifier.add(Dropout(rate=0.2))
        self.classifier.add(Dense((4096)))
        self.classifier.add(ReLU())

        self.classifier.add(Dropout(rate=0.2))
        self.classifier.add(Dense(4096))
        self.classifier.add(ReLU())

        self.classifier.add(Dense(num_classes, activation=tf.nn.softmax))
        
    # @tf.function
    def call(self, x):
        # print('Input shape:', x.get_shape())
        x = tf.reshape(x, (-1, int(cfg.views), int(cfg.img_size), int(cfg.img_size),3))
        # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
        views = tf.transpose(x, perm=[1, 0, 2, 3, 4])
        
        view_pool = tf.TensorArray(dtype=tf.float32, size=int(cfg.views))
        for i in range(int(cfg.views)):
            v = views[i]
            v = tf.reshape(v, (-1, int(cfg.img_size), int(cfg.img_size),3))
            v = self.features(v)
            view_pool = view_pool.write(i, v)
        view_pool = view_pool.stack()
        
        # View pooling
        pooled_view = tf.expand_dims(view_pool[0], 0) # eg. [100] -> [1, 100]
        for i in range(1, int(cfg.views)):
            v = tf.expand_dims(view_pool[i], 0)
            pooled_view = tf.concat([pooled_view, v], 0)
        # print('Pooled view before reducing:', pooled_view.get_shape().as_list())
        
        # Max pooling
        pooled_view = tf.reduce_max(pooled_view, [0], name="view_pool")
        
        # Average pooling
        # pooled_view = tf.reduce_mean(pooled_view, [0], name="view_pool")

        y = self.classifier(pooled_view)
        # print('Output shape:', y.get_shape())
        return y
    
def mvcnn(pretrained=False, pretrained_resnet=True, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MVCNN(**kwargs)
    input_shape = (None, cfg.views, cfg.img_size, cfg.img_size, 3)
    model.build(input_shape=input_shape)
    
    if pretrained:
        model = tf.keras.models.load_model(pretrained)
        
    print("Model initiated - MVCNN")
    return model
