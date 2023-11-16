from torchvision.transforms import transforms
from models.model import resnet_model


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
    transforms.ToPILImage(),
])


image_name = 'dog'
image_path = f'./images/{image_name}.png'
image_save = f'./images/{image_name}_adv.png'

model = resnet_model
model.eval()
