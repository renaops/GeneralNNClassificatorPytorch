import torch
import argparse
from inf721_model import GenericClassificationNet
from inf721_helpers import custom_transforms
from PIL import Image


def load_class_labels(class_labels_file):
    """
    Load class labels from a text file.

    Args:
        class_labels_file (str): Path to the text file containing class labels.

    Returns:
        list: List of class labels.
    """
    with open(class_labels_file, 'r') as file:
        class_labels = [line.strip() for line in file.readlines()]
    return class_labels


def load_image(image_path):
    """
    Load and preprocess the input image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed input image as a PyTorch tensor.
    """
    transf = custom_transforms()
    input_tensor = transf(Image.open(image_path).convert('RGB'))

    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch


def predict_image(model, image_path, class_labels):
    """
    Perform image classification using the given model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        image_path (str): Path to the input image.
        class_labels (list): List of class labels used during training.

    Returns:
        str: Predicted class label.
    """
    input_batch = load_image(image_path)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        model.cuda()
        input_batch.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    predicted_label = class_labels[predicted_class]

    return predicted_label


def main():
    """
    Main function to perform image classification using a trained model.
    """
    parser = argparse.ArgumentParser(
        description='Image Classification Inference Script')
    parser.add_argument('--IMAGE_PATH', type=str, help='Path to the input image')
    parser.add_argument('--MODEL_PATH', type=str,
                        help='Path to the trained model file')
    parser.add_argument('--CLASS_LABELS_FILE', type=str, default='class_labels.txt',
                        help='Path to the text file containing class labels (default: class_labels.txt)')
    args = parser.parse_args()

    class_labels = load_class_labels(args.CLASS_LABELS_FILE)

    model = GenericClassificationNet(len(class_labels))
    model.load_state_dict(torch.load(args.MODEL_PATH))

    prediction = predict_image(model, args.IMAGE_PATH, class_labels)

    print(f'The predicted class is: {prediction}')


if __name__ == '__main__':
    main()
