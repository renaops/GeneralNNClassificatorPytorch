import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2 as transforms_v2


def get_content_from_partition(partition, data_path):
    """
    Retrieve a list of dictionaries containing information about images in a specified partition.

    Args:
    - partition (str): The partition name, such as 'train', 'test', or 'validation'.
    - data_path (str): The base path where the data is stored.

    Returns:
    list: A list of dictionaries, where each dictionary represents an image with the following keys:
        - 'class': The class or label of the image.
        - 'img': The full path to the image file.

    Example:
    >>> data = get_content_from_partition('train', '/path/to/data')
    >>> print(data)
    [{'class': 'cat', 'img': '/path/to/data/train/cat/cat_image.jpg'},
        {'class': 'dog', 'img': '/path/to/data/train/dog/dog_image.png'},
        ...]
    """
    partition_data = []
    for clazz in os.listdir(os.path.join(data_path, partition)):
        for img in os.listdir(os.path.join(data_path, partition, clazz)):
            if img.endswith('.jpg') or img.endswith('.png'):
                partition_data.append(
                    {'class': clazz, 'img': os.path.join(data_path, partition, clazz, img)})

    return partition_data


def plot_learning_curve(train_losses, val_losses, num_epochs, size=(8, 5)):
    """
    Plot the learning curve based on training and validation losses over epochs.

    Args:
    - train_losses (list): List of training losses for each epoch.
    - val_losses (list): List of validation losses for each epoch.
    - num_epochs (int): Total number of training epochs.
    - size (tuple, optional): Figure size (width, height). Default is (8, 5).

    Returns:
    None

    Example:
    >>> train_loss = [0.1, 0.08, 0.05, 0.04]
    >>> val_loss = [0.2, 0.15, 0.1, 0.08]
    >>> plot_learning_curve(train_loss, val_loss, num_epochs=4)
    """
    plt.figure(figsize=size)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o',
             linestyle='-', color='#ff5a7d', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='x',
             linestyle='-', color='#ff9e00', label='Validation Loss')

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()


def plot_confusion_matrix(conf_matrix, class_labels, title):
    """
    Plot a confusion matrix.

    Args:
    - conf_matrix (array-like): Confusion matrix.
    - class_labels (list): List of class labels.
    - title (str): Title for the plot.

    Returns:
    None

    Example:
    >>> confusion_matrix = [[20, 5], [2, 30]]
    >>> class_labels = ['Class A', 'Class B']
    >>> plot_confusion_matrix(confusion_matrix, class_labels, title='Model Performance')
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                yticklabels=class_labels, xticklabels=class_labels)
    plt.title(f'Confusion Matrix {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_class_distribution(dataframe, column_name, title):
    """
    Plot the distribution of classes in a specified column of a DataFrame.

    Args:
    - dataframe (pandas.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column representing the classes.
    - title (str): The title for the plot.

    Returns:
    None

    Example:
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> data = {'Class': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C']}
    >>> df = pd.DataFrame(data)
    >>> plot_class_distribution(df, 'Class', title='Class Distribution')
    """
    sns.countplot(x=column_name, data=dataframe,
                  order=sorted(dataframe[column_name].unique()))
    plt.title(f'{title} Distribution')
    plt.xticks(rotation=45)
    plt.show()


def custom_transforms(new_size=(224, 224), is_training=False):
    """
    Returns a composed set of image transformations using torchvision.transforms.

    The transformations include:
    - Convert the image to dtype torch.float32 with scaling.
    - Resize the image to (224, 224) with antialiasing.
    - Apply random rotation to the image within the range of -45 to 45 degrees.
    - Randomly flip the image horizontally with a probability of 0.5.

    Returns:
    torchvision.transforms.Compose: Composed set of image transformations.

    Example:
    >>> transform = my_transforms()
    >>> img_transformed = transform(img)
    """

    layers = [transforms_v2.ToImage(),
              transforms_v2.ToDtype(torch.float32, scale=True),
              transforms_v2.Resize(size=new_size, antialias=True)]

    if is_training:
        layers.append(transforms_v2.RandomRotation(degrees=(-45, 45)))
        layers.append(transforms_v2.RandomHorizontalFlip(p=0.5))

    return transforms_v2.Compose(layers)


def format_time(seconds):
    if seconds < 60:
        return f'{seconds:.2f}s'
    else:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return f'{int(m)}m{s:.2f}s' if h <= 0 else f'{h}h{int(m)}m{s:.2f}s'
