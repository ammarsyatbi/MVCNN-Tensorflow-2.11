# MVCNN-Tensorflow-2.11
Converting MVCNN from pytorch to Tensorflow 2.11. Transformation is based on this [repo](https://github.com/RBirkeland/MVCNN-PyTorch).

Dataset structure should be as following:
```
./
└── data/
    ├── train/
    │   └── class/
    │       └── mv1/
    │           ├── img1.jpg
    │           ├── img2.jpg
    │           └── imgn.jpg
    └── test/
        └── class/
            └── mv1/
                ├── img1.jpg
                ├── img2.jpg
                └── imgn.jpg
```

Train
```
python train.py --train_dir path/to/train/dataset --model mobilenet --epochs 10
```

Predict
```
python predict.py --model mobilenet --model_weights path/to/model/weights --mvs path/to/image/folder
```

```evaluate.py``` script haven not been tested. Try at your own risk.

Dockerfile follows SageMaker requirements. This is to allow TrainingJob can be called from SageMaker estimator.

# Reference
- https://github.com/RBirkeland/MVCNN-PyTorch