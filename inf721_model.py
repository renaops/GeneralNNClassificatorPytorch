import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GenericClassificationNet(nn.Module):
    """
    Convolutional Neural Network (CNN) for environmental classification.

    Args:
    - num_classes (int): Number of classes for classification.

    Attributes:
    - pool (nn.MaxPool2d): Max pooling layer.
    - conv1, conv2, conv3, conv4, conv5, conv6 (nn.Conv2d): Convolutional layers.
    - batch1, batch2, batch3, batch4, batch5, batch6, batch7 (nn.BatchNorm2d or nn.BatchNorm1d): Batch normalization layers.
    - dropout1, dropout2, dropout3, dropout4 (nn.Dropout): Dropout layers.
    - fc1 (nn.Linear): Fully connected layer.
    - out (nn.Linear): Output layer.

    Methods:
    - forward: Defines the forward pass of the network.
    """

    def __init__(self, num_classes):
        super(GenericClassificationNet, self).__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.batch4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(64, 128, 3)
        self.batch5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.batch6 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 24 * 24, 128)
        self.batch7 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = F.relu(self.conv4(x))
        x = self.batch4(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.batch5(x)
        x = F.relu(self.conv6(x))
        x = self.batch6(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 24 * 24)

        x = F.relu(self.fc1(x))
        x = self.batch7(x)
        x = self.dropout4(x)
        x = self.out(x)

        return x
    
def calc_val_loss(model, dataloader, loss_function):
    """
    Calculate the average validation loss for a given model, dataloader, and loss function.

    Args:
    - model (torch.nn.Module): The neural network model.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - loss_function (torch.nn.modules.loss._Loss): The loss function used for evaluation.

    Returns:
    float: The average validation loss.

    Example:
    >>> net = GenericClassificationNet(num_classes=10)
    >>> val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    >>> criterion = nn.CrossEntropyLoss()
    >>> val_loss = calc_val_loss(net, val_dataloader, criterion)
    >>> print(val_loss)
    0.1234
    """

    USING_CUDA = torch.cuda.is_available()

    with torch.no_grad():
        total_loss = 0.0
        for batch in dataloader:
            imgs, labels = batch

            if USING_CUDA:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')

            outputs = model(imgs)
            batch_loss = loss_function(outputs, labels).item()
            total_loss += batch_loss

        average_loss = total_loss / len(dataloader)

    return average_loss

def optimize(model, train_loader, val_loader, loss_func, optimizer, num_epochs, output_path):
    """
    Train and optimize a neural network model.

    Args:
    - model (torch.nn.Module): The neural network model.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - loss_func (torch.nn.modules.loss._Loss): The loss function used for training.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - num_epochs (int): Number of training epochs.
    - output_path (str): File path to save the trained model.

    Returns:
    tuple: Lists of training and validation losses over epochs.

    Example:
    >>> net = GenericClassificationNet(num_classes=10)
    >>> train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    >>> val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    >>> criterion = nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    >>> train_losses, val_losses = optimize(net, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10, output_path='model.pth')
    """
    USING_CUDA = torch.cuda.is_available()
    
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):

        total_time = 0
        total_loss = 0.0

        print(f'Epoch: {epoch}/{num_epochs}')
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()
            imgs, labels = batch

            if USING_CUDA:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')

            loss = loss_func(model(imgs), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iteration_time = (time.time() - start_time)
            total_time += iteration_time

            completion_percentage = (i + 1) / len(train_loader) * 100
            formatted_completion_percentage = math.floor(completion_percentage / 3.333)

            print(f'\r{i+1}/{len(train_loader)} [{"="*(formatted_completion_percentage)}{"."*(30 - formatted_completion_percentage)}] - {format_time(iteration_time)}/step - Loss: {loss:.6f}', end='', flush=True)

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(calc_val_loss(model, val_loader, loss_func))
        print(f" || Train Loss:{train_losses[-1]:.6f} - Val Loss:{val_losses[-1]:.6f}")

    print('\nComplete! =)')
    torch.save(model.state_dict(), output_path)

    return train_losses, val_losses