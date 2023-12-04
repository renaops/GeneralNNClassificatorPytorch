import argparse

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from inf721_model import GenericClassificationNet, optimize
from torch.utils.data import DataLoader

from inf721_helpers import get_content_from_partition, custom_transforms, plot_learning_curve
from inf721_dataset import GenericDataset


def main():
    """
    Script for training a model with specified parameters.

    Usage:
        python script_name.py --BATCH_SIZE <batch_size> --NUM_EPOCHS <num_epochs> --NUM_CLASSES <num_classes> --MODEL_SAVE_PATH <save_path> --MODEL_NAME <model_name> --DATASET_NAME <dataset_name>

    Arguments:
        --BATCH_SIZE (int): Batch size for training. Default is 170.
        --NUM_EPOCHS (int): Number of epochs for training. Default is 50.
        --NUM_CLASSES (int): Number of classes in the dataset.
        --MODEL_SAVE_PATH (str): Path to save the trained model. Default is './'.
        --MODEL_NAME (str): Name of the model. Default is 'fodo-101_classification'.
        --DATASET_NAME (str): Name of the dataset.

    Example:
        python script_name.py --BATCH_SIZE 256 --NUM_EPOCHS 100 --NUM_CLASSES 10 --MODEL_SAVE_PATH './models/' --MODEL_NAME 'my_model' --DATASET_NAME 'food-101'
    """

    parser = argparse.ArgumentParser(description='Script with command line arguments')
    
    parser.add_argument('--BATCH_SIZE', type=int, default=170, help='Batch size for training')
    parser.add_argument('--NUM_EPOCHS', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--NUM_CLASSES', type=int, help='Number of classes in the dataset')
    parser.add_argument('--MODEL_SAVE_PATH', default='./', help='Path to save the trained model')
    parser.add_argument('--MODEL_NAME', default='fodo-101_classification', help='Name of the model')
    parser.add_argument('--DATASET_NAME', help='Name of the dataset')
    parser.add_argument('--DATASET_PATH', help='Path of the dataset')

    args = parser.parse_args()

    BATCH_SIZE = args.BATCH_SIZE
    NUM_EPOCHS = args.NUM_EPOCHS
    NUM_CLASSES = args.NUM_CLASSES
    MODEL_SAVE_PATH = args.MODEL_SAVE_PATH
    MODEL_NAME = args.MODEL_NAME
    DATASET_NAME = args.DATASET_NAME
    DATASET_PATH = args.DATASET_PATH

    torch.manual_seed(1)
    USING_CUDA = torch.cuda.is_available()
    f'Pytorch {"" if USING_CUDA else "não "}está usando o CUDA!'

    train_df = pd.DataFrame(get_content_from_partition('train', DATASET_PATH))
    val_df = pd.DataFrame(get_content_from_partition('validation', DATASET_PATH))
    test_df = pd.DataFrame(get_content_from_partition('test', DATASET_PATH))

    print(f'> Train size: {len(train_df)}\n> Val size: {len(val_df)}\n> Test size: {len(test_df)}')

    train_dataset = GenericDataset(train_df, transform=custom_transforms(is_training=True))
    val_dataset = GenericDataset(val_df, transform=custom_transforms())
    test_dataset = GenericDataset(test_df, transform=custom_transforms())

    workers = os.cpu_count()
    print(f"Thread workers: {workers}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

    model = GenericClassificationNet(NUM_CLASSES)

    if USING_CUDA:
        model.cuda()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = optimize(model, train_loader, val_loader, loss_func, optimizer, NUM_EPOCHS, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '.pth'))
    plot_learning_curve(train_losses, val_losses, NUM_EPOCHS)

if __name__ == "__main__":
    main()
