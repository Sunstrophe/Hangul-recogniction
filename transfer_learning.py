import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from HGDataset import HGImageDataset

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_dataset = HGImageDataset(img_dir='data2/hangul_characters_v1',
                               transform=transform)
test_dataset = HGImageDataset(
    img_dir='data2/hangul_characters_v1', transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

num_classes = train_dataset.get_len_unique_labels()

resnet50 = models.resnet50(weights="DEFAULT")
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')


train_model(resnet50, train_loader, criterion, optimizer, num_epochs=10)


def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * total_correct / total_images}%')


evaluate_model(resnet50, test_loader)
