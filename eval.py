import torch
from PIL import Image
import numpy as np
from set_up import image_path, image_save, model, transform
import json


save_dir = 'CNN_model'
save_path = f'{save_dir}/model_1.pth'


with open('models/image_class_label.json', 'r') as f:
    imagenet_labels = json.load(f)

out = model(transform(Image.open(image_path).convert('RGB')).unsqueeze(0))
_, predicted = torch.max(out.data, 1)
print(f'origin: {predicted} ', imagenet_labels[f'{np.array(predicted)[0]}'])

out = model(transform(Image.open(image_save).convert('RGB')).unsqueeze(0))
_, predicted = torch.max(out.data, 1)
print(f'adv: {predicted} ', imagenet_labels[f'{np.array(predicted)[0]}'])