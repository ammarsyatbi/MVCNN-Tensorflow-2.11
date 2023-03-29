import tensorflow as tf

print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    ReLU,
    Dropout
)
from tensorflow.keras import Sequential
from config import cfg
class MobileNet(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        # Base Model
        self.features = tf.keras.applications.MobileNetV3Large(
            input_shape=(int(cfg.img_size), int(cfg.img_size),3),
            alpha=1.0,
            include_top=False,
            weights='imagenet',
            dropout_rate=0.2,
        )
        self.flatter = Flatten()
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
            v = self.flatter(v)
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
    
def mobilenet(pretrained=False, **kwargs):
    r"""MobileNetV3 base model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet(**kwargs)
    if pretrained:
        model = tf.keras.models.load_model(pretrained)
    print("Model initiated - MobileNetV3")
    
    return model
