import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")