#include "webcam_capture.h"
#include "torch_inference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// ImageNet class names (first few for demonstration)
std::vector<std::string> imagenet_classes = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
    // ... (you can add more or load from file)
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light"
    // Note: This is just a sample. Full ImageNet has 1000 classes
};

std::string getTopPrediction(const torch::Tensor &output)
{
    // Apply softmax to get probabilities
    auto probs = torch::softmax(output, 1);

    // Get the index of the maximum value
    auto max_result = torch::max(probs, 1);
    int predicted_class = std::get<1>(max_result).item<int>();
    float confidence = std::get<0>(max_result).item<float>();

    std::string class_name = (predicted_class < imagenet_classes.size())
                                 ? imagenet_classes[predicted_class]
                                 : "class_" + std::to_string(predicted_class);

    return class_name + " (" + std::to_string(confidence * 100).substr(0, 5) + "%)";
}

int main()
{
    // Initialize webcam
    WebcamCapture webcam;
    if (!webcam.initialize(0))
    {
        std::cerr << "Failed to initialize webcam" << std::endl;
        return -1;
    }

    // Initialize model inference
    TorchInference inference;

    // Try to load model
    std::string model_path = "model.pt";
    if (!inference.loadModel(model_path))
    {
        std::cerr << "Failed to load model, continuing without inference..." << std::endl;
    }

    // Set device (try CUDA first, fallback to CPU)
    inference.setDevice("cuda");
    inference.printModelInfo();

    std::cout << "Press 'q' to quit, 's' to save frame, 'i' to run inference" << std::endl;
    std::cout << "ResNet18 will classify objects in the webcam feed!" << std::endl;

    int frame_count = 0;
    while (true)
    {
        cv::Mat frame = webcam.captureFrame();

        if (frame.empty())
        {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }

        // Display frame
        cv::imshow("Webcam with ResNet18 Classification", frame);

        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
        { // 'q' or ESC
            break;
        }
        else if (key == 's')
        {
            std::string filename = "capture_" + std::to_string(frame_count++) + ".jpg";
            if (webcam.saveFrame(frame, filename))
            {
                std::cout << "Frame saved as " << filename << std::endl;
            }
        }
        else if (key == 'i' && inference.isModelLoaded())
        {
            try
            {
                std::cout << "Running ResNet18 classification..." << std::endl;

                // Run inference on current frame
                torch::Tensor output = inference.predictFromImage(frame);

                // Get top prediction
                std::string prediction = getTopPrediction(output);
                std::cout << "Prediction: " << prediction << std::endl;

                // Also show raw output info
                std::cout << "Output shape: " << output.sizes() << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Inference error: " << e.what() << std::endl;
            }
        }
    }

    webcam.stopCapture();
    cv::destroyAllWindows();

    return 0;
}