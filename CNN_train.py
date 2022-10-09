import copy

import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import torch.nn as nn
import os
from PIL import Image

from myproject01_dogcat import CNN_model

datafolder = os.path.join(".//train")
pictures = os.listdir(datafolder)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(250),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = []

print("generating datasets...")
for picture in pictures:
    picture_dir = datafolder + "//" + picture
    # print(picture_dir)
    image = Image.open(picture_dir)
    image_transformed = transform(image)
    # 规定 0 就是猫， 1 就是狗
    if picture[0] == "c":
        label = 0
    else:
        label = 1
    dataset.append([image_transformed, label])
    # dataset.append([0, 0])

train_size = int(0.75*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print("dataset generation complete: " + str(train_size) + " train samples, " + str(val_size) + " validation samples")

dataloaders = {}
dataloaders["train"] = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
dataloaders["val"] = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)
print("dataloader generation complete")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model(model, optimizer, criterian, scheduler, epoches):
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(epoches):
        print("=== epoch", epoch+1, "of", epoches, "===")
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for input, label in dataloaders[phase]:
                input = input.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(input)
                    loss = criterian(outputs, label)
                    index, pred = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input.size(0)
                running_corrects += torch.sum(pred == label.data)


            epoch_loss = running_loss / (train_size if phase=="train" else val_size)
            epoch_acc = running_corrects.double() / (train_size if phase == "train" else val_size)

            print('本次epoch {} 结果：Loss: {:.4f}, Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
        print()
    print("The best accuracy is {:4f}".format(best_acc))
    model.load_state_dict(best_model)
    return model


model = CNN_model.CNN_model()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterian = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, optimizer, criterian, scheduler, 20)
torch.save(model.state_dict(), ".//CNNmodel01.pth")



