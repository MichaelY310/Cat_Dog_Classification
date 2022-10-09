import torch
from torchvision import transforms, models
import torch.nn as nn
import CNN_model
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择模型和图片
# # CNN Model
# model = CNN_model.CNN_model()
# model.load_state_dict(torch.load('./CNNmodel01.pth'))

# Resnet Model
model = models.resnet18()
feature_number = model.fc.in_features
model.fc = nn.Linear(feature_number, 2)
model = model.to(device)
model.load_state_dict(torch.load('./Resnetmodel01.pth'))

picture_location = ".//test//cat01.jpg"




image = Image.open(picture_location)
if image.size[0] > image.size[1]:
    image = image.resize((250 * image.size[0] // image.size[1], 250))
else:
    image = image.resize((250, 250 * image.size[1] // image.size[0]))

transform = transforms.Compose([
    transforms.RandomCrop(250),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_transformed = transform(image)
print(image_transformed.size())
list = [image_transformed.numpy().tolist()]
dataset = torch.Tensor(list)
dataset = dataset.to(device)
print(dataset.size())
output = model(dataset)
print(output)

# 0: cat
# 1: dog
Truelabel = torch.tensor([1], dtype=torch.long)
Falselabel = torch.tensor([0], dtype=torch.long)
Truelabel = Truelabel.to(device)
Falselabel = Falselabel.to(device)

criterian = nn.CrossEntropyLoss()
Trueloss = criterian(output, Truelabel)
print(Trueloss.item())
Falseloss = criterian(output, Falselabel)
print(Falseloss.item())

if output[0][0] > output[0][1]:
    print("this is a cat")
else:
    print("this is a dog")


# torch.Size([4, 3, 250, 250])
# <class 'torch.Tensor'>

# tensor([[ 7.5569, -8.1957]], grad_fn=<AddmmBackward0>)
# tensor([[-4.9077,  5.5237]], grad_fn=<AddmmBackward0>)