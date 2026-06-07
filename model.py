from glob import glob
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

classes = glob('Train/Full moon/*')
print(classes)

print(os.getcwd())