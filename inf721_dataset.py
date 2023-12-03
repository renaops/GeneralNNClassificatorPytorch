class GenericDataset(Dataset):
    """
    A custom PyTorch dataset for working with image data in environments.

    Args:
    - df (pandas.DataFrame): The DataFrame containing image file paths and class labels.
    - transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
    - target_name (str, optional): The name of the column in 'df' containing class labels. Default is 'class'.
    - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

    Attributes:
    - class_names (list): List of unique class names present in the dataset.
    - data (pandas.DataFrame): The DataFrame representing the dataset.

    Methods:
    - __len__: Returns the number of samples in the dataset.
    - __getitem__: Returns the image and label for a given index.

    Example:
    >>> dataset = GenericDataset(df, transform=transforms.ToTensor(), target_name='label', shuffle=True)
    >>> print(dataset.class_names)
    ['class_A', 'class_B', 'class_C']
    >>> print(len(dataset))
    1000
    >>> img, label = dataset[0]
    >>> print(img.shape, label)
    (3, 224, 224), 0
    """

    def __init__(self, df, transform=None, target_name='class', shuffle=True):
        self._data = self._shuffle(df) if shuffle else df
        self._transform = transform
        self._class_names = sorted(list(self.data[target_name].unique()))

    @property
    def class_names(self):
        """List of unique class names present in the dataset."""
        return self._class_names

    @property
    def data(self):
        """The DataFrame representing the dataset."""
        return self._data

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the image and label for a given index."""
        img = read_image(self._data.loc[idx, 'img'])[:3, :, :]

        label = self._class_names.index(self._data.loc[idx, 'class'])

        if self._transform:
            img = self._transform(img)

        return img, label

    def _shuffle(self, df):
        """Shuffles the DataFrame."""
        return df.sample(frac=1).reset_index(drop=True)
