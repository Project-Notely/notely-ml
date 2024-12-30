#!/bin/sh

if [ "$1" == "--train" ]; then
    PYTHONPATH=. python scripts/train_emnist_cnn.py
elif [ "$1" == "--predict" ]; then
    PYTHONPATH=. python scripts/predict.py --model_path ./checkpoints/emnist_cnn_best.pth --image_path data/a.png
else
    echo "Invalid argument"
fi