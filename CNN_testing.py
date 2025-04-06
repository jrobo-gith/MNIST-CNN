import torch
import torchvision.transforms as transforms
import torchvision
from CNN_structure import CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}.")

batch_size = 100

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
model = CNN().to(device)
model.load_state_dict(torch.load('model/cnn_weights.pth', map_location=device))


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))