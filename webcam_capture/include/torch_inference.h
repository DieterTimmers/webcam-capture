#ifndef TORCH_INFERENCE_H
#define TORCH_INFERENCE_H

#include <torch/torch.h>
#include <torch/script.h> // Add this for JIT support
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class TorchInference
{
public:
    TorchInference();
    ~TorchInference();

    // Load a PyTorch model from .pt file
    bool loadModel(const std::string &model_path);

    // Check if model is loaded
    bool isModelLoaded() const;

    // Inference methods
    torch::Tensor predict(const torch::Tensor &input);
    torch::Tensor predictFromImage(const cv::Mat &image);
    std::vector<float> predictToVector(const torch::Tensor &input);

    // Utility functions for OpenCV <-> Torch conversion
    torch::Tensor matToTensor(const cv::Mat &mat, bool normalize = true);
    cv::Mat tensorToMat(const torch::Tensor &tensor);

    // Model info
    void printModelInfo();

    // Set device (CPU/CUDA)
    void setDevice(const std::string &device = "cpu");
    torch::Device getDevice() const;

private:
    torch::jit::script::Module model_;
    bool model_loaded_;
    torch::Device device_;
    bool is_jit_model_;

    // Helper functions
    torch::Tensor preprocessImage(const cv::Mat &image);
    cv::Mat postprocessTensor(const torch::Tensor &tensor);
};

#endif // TORCH_INFERENCE_H