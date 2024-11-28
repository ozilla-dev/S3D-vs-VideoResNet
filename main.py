import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd

class JestureDataset(Dataset):
    def __init__(self, data, data_path):
        self.data = data
        self.data_path = data_path

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        video_id, label = self.data.iloc[i]
        video_folder = os.path.join(self.data_path, str(video_id))
        video_tensor = self.load_videos(video_folder)
        return video_tensor, label

    def load_videos(self, video_folder):
        frames = []
        for frame in os.listdir(video_folder):
            frames.append(read_image(os.path.join(video_folder, frame)))
        return torch.stack(frames)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_label_encoding(labels_path):
    labels_df = pd.read_csv(labels_path, header=None)
    label_encoding = {label: idx for idx, label in enumerate(labels_df[0])}
    return label_encoding

def split_data(labels_path):
    label_encoding = create_label_encoding(labels_path)
    
    train_data = pd.read_csv('jester-v1-train.csv', sep=';', header=None)
    test_data = pd.read_csv('jester-v1-validation.csv', sep=';', header=None)

    # Reduce the size of the datasets by a factor of 10

    val_data = train_data.sample(frac=0.1, random_state=42)
    train_data = train_data.drop(val_data.index) # remove validation data from training data
    train_data[1] = train_data[1].map(label_encoding)
    val_data[1] = val_data[1].map(label_encoding)
    test_data[1] = test_data[1].map(label_encoding)
    
    return train_data, val_data, test_data

def main():
    seed = 42
    set_seed(seed)

    data_path = r'data\20bnjester-v1\20bn-jester-v1'
    labels_path = 'jester-v1-labels.csv'

    train_data, val_data, test_data = split_data(labels_path)
    print(f"Train length: {len(train_data)}")
    print(f"Validation length: {len(val_data)}")
    print(f"Test length: {len(test_data)}")

    train_dataset = JestureDataset(train_data, data_path)
    val_dataset = JestureDataset(val_data, data_path)
    test_dataset = JestureDataset(test_data, data_path)

    print(len(train_dataset))
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(val_dataset, 'val_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')

if __name__ == '__main__':
    main()
