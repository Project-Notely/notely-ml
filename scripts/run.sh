#!/bin/sh

if [ "$1" == "--train" ]; then
    PYTHONPATH=. python scripts/train_emnist_cnn.py
elif [ "$1" == "--predict" ]; then
    PYTHONPATH=. python scripts/predict.py --model_path ./trained_models/emnist_cnn_byclass.pth --image_path data/5.png
else
    echo "Invalid argument"
fi