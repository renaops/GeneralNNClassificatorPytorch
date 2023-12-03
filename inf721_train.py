import argparse

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

    args = parser.parse_args()


    BATCH_SIZE = args.BATCH_SIZE
    NUM_EPOCHS = args.NUM_EPOCHS
    NUM_CLASSES = args.NUM_CLASSES
    MODEL_SAVE_PATH = args.MODEL_SAVE_PATH
    MODEL_NAME = args.MODEL_NAME
    DATASET_NAME = args.DATASET_NAME

if __name__ == "__main__":
    main()
