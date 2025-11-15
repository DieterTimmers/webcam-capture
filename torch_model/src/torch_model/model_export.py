import torch
import torchvision

# Use ResNet18 - a simple classification model that works with standard LibTorch
model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()

# Create a dummy input to trace the model (typical webcam resolution)
dummy_input = torch.randn(1, 3, 224, 224)  # ResNet expects 224x224

# Use torch.jit.trace instead of torch.jit.script for better compatibility
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model.pt")

print("ResNet18 model saved as model.pt")
print("This model is compatible with standard LibTorch C++!")
