import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import Resize
import os
import pandas as pd

class JesterDataset(Dataset):
    def __init__(self, data, data_path, num_frames=37, frame_size=(100, 176)):
        self.data = data
        self.data_path = data_path
        self.num_frames = num_frames
        self.resize = Resize(frame_size)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        video_id, label = self.data.iloc[i]
        video_folder = os.path.join(self.data_path, str(video_id))
        video_tensor = self.load_videos(video_folder)
        return video_tensor, label

    def load_videos(self, video_folder):
        frames = []
        for frame in sorted(os.listdir(video_folder)):
            original_frame = read_image(os.path.join(video_folder, frame))
            frame = self.resize(original_frame)
            frames.append(frame)
        if len(frames) < self.num_frames:
            num_missing_frames = self.num_frames - len(frames)
            # take last frame and duplicate it to fill in missing frames
            frames.extend([frames[-1]] * num_missing_frames)
        elif len(frames) > self.num_frames:
            # uniformly choose frames to keep the features
            indices = torch.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        video_tensor = torch.stack(frames).float() # convert to float because of mismatch with pretrained model weights
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor

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

    train_data = train_data.sample(frac=0.005, random_state=42)
    val_data = train_data.sample(frac=0.1, random_state=42)
    test_data = test_data.sample(frac=0.005, random_state=42)
    train_data = train_data.drop(val_data.index) # remove validation data from training data
    train_data[1] = train_data[1].map(label_encoding)
    val_data[1] = val_data[1].map(label_encoding)
    test_data[1] = test_data[1].map(label_encoding)

    print("Unique labels in training data:", sorted(train_data[1].unique()))
    print("Unique labels in validation data:", val_data[1].unique())
    print("Unique labels in test data:", test_data[1].unique())
    
    
    return train_data, val_data, test_data

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        for video, label in train_loader:
            video = video.to(device)
            label = label.to(device)
            label_pred = model(video)
            loss = criterion(label_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
        model.train()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_{epoch + 1}.pth')

def validate(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for video, label in val_loader:
            video = video.to(device)
            label = label.to(device)
            label_pred = model(video)
            loss = criterion(label_pred, label)
            total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
    return avg_loss

def main():
    seed = 42
    set_seed(seed)

    data_path = r'data\20bnjester-v1\20bn-jester-v1'
    labels_path = 'jester-v1-labels.csv'

    train_data, val_data, test_data = split_data(labels_path)
    print(f"Train length: {len(train_data)}")
    print(f"Validation length: {len(val_data)}")
    print(f"Test length: {len(test_data)}")

    train_dataset = JesterDataset(train_data, data_path)
    val_dataset = JesterDataset(val_data, data_path)
    test_dataset = JesterDataset(test_data, data_path)

    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(val_dataset, 'val_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')

    batch_size = 32
    num_workers = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    
    output_size = len(set(train_data[1]))
    # change the output size of the pretrained model to match the number of classes in my dataset
    model.fc = torch.nn.Linear(model.fc.in_features, output_size)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()
