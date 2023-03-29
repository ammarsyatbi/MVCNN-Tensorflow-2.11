# MVCNN-Tensorflow-2.11
Converting MVCNN from pytorch to Tensorflow 2.11. Transformation is based on this [repo](https://github.com/RBirkeland/MVCNN-PyTorch).

Train
```
python train.py --model mobilenet --epochs 10
```

Predict
```
python predict.py --model mobilenet --model_weights path/to/model/weights --mvs path/to/image/folder
```

```evaluate.py``` script haven't tested

# Reference
- https://github.com/RBirkeland/MVCNN-PyTorch