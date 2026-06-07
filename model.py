import os
from glob import glob
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

classes = glob('Train/Full moon/*')
print(classes)

print(os.getcwd())

file_path='Train/Full moon'
for img_name in os.listdir(file_path):
    img=Image.open(os.path.join(file_path,img_name))
    sizes=(set(img.size))

#for image auhmentatiion 
train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((140,140)),
    transforms.CenterCrop(128),
    transforms.RandomRotation(degrees=360),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#doing this for checking if the directory is correct 
train_dataset = datasets.ImageFolder(
    root='Train',              
    transform=train_transforms
)

train_loader=DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
print(f'Total images: {len(train_dataset)}')
print(f'Classes found: {train_dataset.classes}')

## output is 
## Total images: 249
## Classes found: ['First Quarter', 'Full moon', 'Last Quarter', 'New moon'
## above output is from the jupyter notebook file output 