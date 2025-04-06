import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from CNN_structure import CNN

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}.")

model = CNN().to(device)
learning_rate = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

torch.save(model.state_dict(), 'model/cnn_weights.pth')
