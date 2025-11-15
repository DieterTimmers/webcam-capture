import torch
import torchvision

# RetinaNet with ResNet50 backbone
model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
model.eval()

dummy_input = torch.randn(1, 3, 416, 416)

scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

print("RetinaNet ResNet50 model saved as model.pt")
print("- Single-shot detector")
print("- Good for real-time applications")
