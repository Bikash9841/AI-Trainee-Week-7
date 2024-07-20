
from alexnet_model import Alexnet
import torch
from torchvision import transforms


alexnet = Alexnet()
alexnet.load_state_dict(torch.load(
    '/home/bikasherl/Desktop/Week 7/apps/alexnetr.pt'))


transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image):
    img = transform(image)
    logits = alexnet(img.unsqueeze(0))
    class_label = logits.argmax(1)
    return class_label.item()
