from CNN_structure import CNN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

model = CNN()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Testing on {device}.")

batch_size = 100

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
model = CNN().to(device)
model.load_state_dict(torch.load('model/cnn_weights.pth', map_location=device))

predictions = []
true_labels = []

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

        predicted = np.array(predicted)
        predictions.append(predicted)

        labels = np.array(labels)
        true_labels.append(labels)

predictions = np.array(predictions).flatten()
true_labels = np.array(true_labels).flatten()

num_images = 16
num_rows = int(np.sqrt(num_images))

rand_ints = np.random.randint(0, len(predictions), size=num_images)

fig, ax = plt.subplots(nrows=num_rows, ncols=num_rows, figsize=(10, 10))
i = 0
for row in range(num_rows):
    for col in range(num_rows):
        image = test_dataset[rand_ints[i]][0].squeeze().numpy()
        ax[row, col].imshow(image, cmap='Greys')
        ax[row, col].set_title(f"Label {predictions[rand_ints[i]]}")
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        if predictions[rand_ints[i]] == true_labels[rand_ints[i]]:
            ax[row, col].set_title(f"Prediction {predictions[rand_ints[i]]}", color='Green')
        else:
            ax[row, col].set_title(f"Label {predictions[rand_ints[i]]}", color='Red')

        i += 1

plt.savefig("figures/_16x16_.png")