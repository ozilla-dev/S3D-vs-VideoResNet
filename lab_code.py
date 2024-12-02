import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')

    batch_size = 32
    num_workers = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    output_size = len(set(train_dataset.data[1]))
    # change the output size of the pretrained model to match the number of classes in my dataset
    model.fc = torch.nn.Linear(model.fc.in_features, output_size)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()
