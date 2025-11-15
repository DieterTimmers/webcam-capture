#include "torch_inference.h"
#include <iostream>
#include <exception>

TorchInference::TorchInference()
    : model_loaded_(false), device_(torch::kCPU), is_jit_model_(false)
{
}

TorchInference::~TorchInference()
{
    // Cleanup is automatic with torch::jit::script::Module
}

bool TorchInference::loadModel(const std::string &model_path)
{
    try
    {
        std::cout << "Loading model from: " << model_path << std::endl;

        // Try to load as JIT model first
        try
        {
            model_ = torch::jit::load(model_path, device_);
            model_.eval(); // Set to evaluation mode
            is_jit_model_ = true;
            model_loaded_ = true;
            std::cout << "Model loaded successfully as JIT model!" << std::endl;
            return true;
        }
        catch (const std::exception &jit_error)
        {
            std::cout << "JIT loading failed: " << jit_error.what() << std::endl;
            std::cout << "Note: JIT models (.pt files from torch.jit.save()) may not be supported in CPU-only LibTorch builds." << std::endl;
            std::cout << "Consider using torch.save() for tensors or upgrading to full LibTorch." << std::endl;
            return false;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

bool TorchInference::isModelLoaded() const
{
    return model_loaded_;
}

torch::Tensor TorchInference::predict(const torch::Tensor &input)
{
    if (!model_loaded_)
    {
        throw std::runtime_error("Model not loaded!");
    }

    try
    {
        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;

        // Move input to the same device as model
        auto input_device = input.to(device_);

        if (is_jit_model_)
        {
            // Run inference with JIT model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_device);

            at::Tensor output = model_.forward(inputs).toTensor();
            return output;
        }
        else
        {
            throw std::runtime_error("Non-JIT model inference not implemented yet!");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw;
    }
}

torch::Tensor TorchInference::predictFromImage(const cv::Mat &image)
{
    if (image.empty())
    {
        throw std::runtime_error("Input image is empty!");
    }

    // Convert image to tensor
    torch::Tensor input_tensor = matToTensor(image, true);

    // Add batch dimension if needed
    if (input_tensor.dim() == 3)
    {
        input_tensor = input_tensor.unsqueeze(0); // Add batch dimension
    }

    return predict(input_tensor);
}

std::vector<float> TorchInference::predictToVector(const torch::Tensor &input)
{
    torch::Tensor output = predict(input);

    // Convert tensor to vector
    output = output.cpu(); // Move to CPU if needed

    std::vector<float> result(output.data_ptr<float>(),
                              output.data_ptr<float>() + output.numel());

    return result;
}

torch::Tensor TorchInference::matToTensor(const cv::Mat &mat, bool normalize)
{
    // Convert BGR to RGB
    cv::Mat rgb_mat;
    if (mat.channels() == 3)
    {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    }
    else
    {
        rgb_mat = mat.clone();
    }

    // Convert to float and normalize to [0, 1]
    rgb_mat.convertTo(rgb_mat, CV_32F);
    if (normalize)
    {
        rgb_mat /= 255.0f;
    }

    // Convert to tensor (HWC -> CHW)
    torch::Tensor tensor = torch::from_blob(
        rgb_mat.data,
        {rgb_mat.rows, rgb_mat.cols, rgb_mat.channels()},
        torch::kFloat);

    // Permute dimensions: HWC -> CHW
    tensor = tensor.permute({2, 0, 1});

    return tensor.clone(); // Make a copy to avoid data corruption
}

cv::Mat TorchInference::tensorToMat(const torch::Tensor &tensor)
{
    torch::Tensor cpu_tensor = tensor.cpu();

    // Handle different tensor shapes
    torch::Tensor processed_tensor;
    if (cpu_tensor.dim() == 4 && cpu_tensor.size(0) == 1)
    {
        // Remove batch dimension
        processed_tensor = cpu_tensor.squeeze(0);
    }
    else if (cpu_tensor.dim() == 3)
    {
        processed_tensor = cpu_tensor;
    }
    else
    {
        throw std::runtime_error("Unsupported tensor shape for conversion to Mat");
    }

    // Permute CHW -> HWC
    processed_tensor = processed_tensor.permute({1, 2, 0});

    // Clamp values to [0, 1] and convert to [0, 255]
    processed_tensor = torch::clamp(processed_tensor, 0.0, 1.0) * 255.0;
    processed_tensor = processed_tensor.to(torch::kUInt8);

    // Create OpenCV Mat
    int height = processed_tensor.size(0);
    int width = processed_tensor.size(1);
    int channels = processed_tensor.size(2);

    cv::Mat result(height, width,
                   channels == 1 ? CV_8UC1 : CV_8UC3,
                   processed_tensor.data_ptr<uint8_t>());

    // Convert RGB to BGR if needed
    if (channels == 3)
    {
        cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    }

    return result.clone();
}

void TorchInference::printModelInfo()
{
    if (!model_loaded_)
    {
        std::cout << "No model loaded." << std::endl;
        return;
    }

    std::cout << "Model Information:" << std::endl;
    std::cout << "Device: " << device_ << std::endl;
    std::cout << "Model type: " << (is_jit_model_ ? "JIT (TorchScript)" : "Regular PyTorch") << std::endl;

    if (is_jit_model_)
    {
        // Print model methods (if any)
        auto methods = model_.get_methods();
        std::cout << "Available methods: ";
        for (const auto &method : methods)
        {
            std::cout << method.name() << " ";
        }
        std::cout << std::endl;
    }
}

void TorchInference::setDevice(const std::string &device)
{
    if (device == "cuda" && torch::cuda::is_available())
    {
        device_ = torch::kCUDA;
        std::cout << "Using CUDA device" << std::endl;
    }
    else if (device == "cpu")
    {
        device_ = torch::kCPU;
        std::cout << "Using CPU device" << std::endl;
    }
    else
    {
        device_ = torch::kCPU;
        std::cout << "CUDA not available, using CPU device" << std::endl;
    }

    // Move model to new device if loaded
    if (model_loaded_ && is_jit_model_)
    {
        model_.to(device_);
    }
}

torch::Device TorchInference::getDevice() const
{
    return device_;
}