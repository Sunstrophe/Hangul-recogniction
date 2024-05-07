import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from HGDataset import HGImageDataset
from datetime import datetime


# Create a dataset object
train_dataset = HGImageDataset(
    img_dir='data2/hangul_characters_v1', transform=transforms.ToTensor())


test_dataset = HGImageDataset(
    img_dir='data2/hangul_characters_v1', transform=transforms.ToTensor(), train=False)

train_data = DataLoader(train_dataset, batch_size=4)
test_data = DataLoader(test_dataset, batch_size=4)

label_len = train_dataset.get_len_unique_labels()

# print(train_data)
# print(len(train_data))
# print(len(test_data))
# for images, labels in train_data:
#     print(type(labels), labels)
#     break


class SimpleCNN(torch.nn.Module):
    def __init__(self, label_len):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(32*14*14, label_len)
        )

    def forward(self, x):
        return self.sequence(x)


model = SimpleCNN(label_len)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, train_data, criterion, optimizer):
    size = len(train_data.dataset)
    model.train()
    for batch, (data, label) in enumerate(train_data):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_model(model, test_data, criterion):
    size = len(test_data.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, label in test_data:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            test_loss += criterion(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start_time = datetime.now()
    train_model(model, train_data, criterion, optimizer)
    test_model(model, test_data, criterion)
    print(f"Time taken: {(datetime.now() - start_time).total_seconds()} seconds\n")

print("Done!")
