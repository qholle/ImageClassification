import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# This function loads the data into a DataLoader object
def get_data_loader(training = True):
    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training == True:
        train_set=datasets.FashionMNIST('./data',train=True,
            download=True,transform=custom_transform)
        return torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)
        return torch.utils.data.DataLoader(test_set, batch_size = 64)

# This function builds the neural network model that we will be using to perform image classification
def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

# This function trains the model and outputs the model's performance throughout training epochs
def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        runningLoss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            runningLoss += loss.item()
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct += 1
                total += 1

        print('Train Epoch: %d Accuracy: %d/%d(%.2f%%) Loss: %.3f' % (epoch, correct, total, (correct/total) * 100, runningLoss / len(train_loader)))

# This function evaluates the model that we have trained, using test set data.
def evaluate_model(model, test_loader, criterion, show_loss = True):
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            correct = 0
            total = 0
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct += 1
                total += 1
    if show_loss == True:
        print('Average loss: %.3f' % (loss.item()))
    print('Accuracy: %.2f%%' % ((correct/total) * 100))

# This function predicts the label of a given image, and gives the top three most likely labels
def predict_label(model, test_images, index):
    outputs = model(test_images)
    prob = F.softmax(outputs, dim=1) # convert to probabilities using softmax to make it easier to read for humans
    labelProbs = {'T-shirt/top': prob[index][0].item(),'Trouser': prob[index][1].item(),'Pullover': prob[index][2].item(),'Dress': prob[index][3].item(),'Coat': prob[index][4].item(),'Sandal': prob[index][5].item(),'Shirt': prob[index][6].item(),'Sneaker': prob[index][7].item(),'Bag': prob[index][8].item(),'Ankle Boot': prob[index][9].item()}
    sortedProbs = sorted(labelProbs.items(), key=lambda x: x[1], reverse=True)
    print('%s: %.2f%%' % (sortedProbs[0][0], sortedProbs[0][1] * 100))
    print('%s: %.2f%%' % (sortedProbs[1][0], sortedProbs[1][1] * 100))
    print('%s: %.2f%%' % (sortedProbs[2][0], sortedProbs[2][1] * 100))

if __name__ == '__main__':
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)

    test_loader = get_data_loader(False)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()
    model.train()
    train_model(model, train_loader, criterion, 5)

    model.eval()
    evaluate_model(model, test_loader, criterion)

    pred_set, _ = next(iter(test_loader))

    # Predicting the label of an image that is a ankle boot
    predict_label(model, pred_set, 0)
    print()

    # Predicting the label of an image that is a pullover
    predict_label(model, pred_set, 1)
    print()

    # Predicting the label of an image that is a trouser
    predict_label(model, pred_set, 2)
    print()

    # Predicting the label of an image that is a trouser
    predict_label(model, pred_set, 3)
    print()

    # Predicting the label of an image that is a shirt
    predict_label(model, pred_set, 4)
