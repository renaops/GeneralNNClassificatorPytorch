import argparse
import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def visualize_class_distribution(df, title):
    sns.set(style="whitegrid")
    sns.set_palette("rocket")
    ax = sns.countplot(x='label', data=df, order=sorted(df['label'].unique()))
    ax.set(title=title)
    plt.xticks(rotation=45, ha="right")
    plt.show()

def copy_images(df, folder, directory_source_path, path_destination):
    print(f"\n{folder.capitalize()}")
    for index, row in df.iterrows():
        print(f"\r{index}/{len(df)}", end="", flush=True)
        file_path_source = os.path.join(directory_source_path, row['image_path'])
        new_file_path = os.path.join(path_destination, folder, row['image_path'])
        if not os.path.exists(new_file_path):
            shutil.copy(file_path_source, new_file_path)

def main():
    parser = argparse.ArgumentParser(description='Script with command line arguments')

    parser.add_argument('--DIRECTORY_SOURCE_PATH', help='Dataser source path')
    parser.add_argument('--PATH_DESTINATION', default='fodo-./', help='Destination path')

    args = parser.parse_args()

    DIRECTORY_SOURCE_PATH = args.DIRECTORY_SOURCE_PATH
    PATH_DESTINATION = args.PATH_DESTINATION

    all_items = os.listdir(DIRECTORY_SOURCE_PATH)
    labels = [item for item in all_items if os.path.isdir(os.path.join(DIRECTORY_SOURCE_PATH, item))]

    print("List of folders/labels:")
    for folder in labels:
        print(folder)

    paths = []
    label_list = []
    for i in range(0, 10):
        print(labels[i])
        for image_name in os.listdir(os.path.join(DIRECTORY_SOURCE_PATH, labels[i])):
            image_path = os.path.join(labels[i], image_name)
            label_list.append(labels[i])
            paths.append(image_path)

    data = pd.DataFrame({'label': label_list, 'image_path': paths})

    train_df, test_df = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, shuffle=True)

    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    val_df = val_df.reset_index()

    visualize_class_distribution(train_df, 'Train Class Distribution')
    visualize_class_distribution(test_df, 'Test Class Distribution')
    visualize_class_distribution(val_df, 'Val Class Distribution')

    target_folders = ["test", "train", "val"]
    for folder in target_folders:
        if not os.path.exists(os.path.join(PATH_DESTINATION, folder)):
            os.makedirs(os.path.join(PATH_DESTINATION, folder))

    for i in range(0, 10):
        for folder in target_folders:
            label_folder_path = os.path.join(PATH_DESTINATION, folder, labels[i])
            if not os.path.exists(label_folder_path):
                os.makedirs(label_folder_path)


    copy_images(train_df, 'train', DIRECTORY_SOURCE_PATH, PATH_DESTINATION)
    copy_images(test_df, 'test', DIRECTORY_SOURCE_PATH, PATH_DESTINATION)
    copy_images(val_df, 'val', DIRECTORY_SOURCE_PATH, PATH_DESTINATION)

    print("\nLengths:")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Val: {len(val_df)}")


if __name__ == "__main__":
    main()