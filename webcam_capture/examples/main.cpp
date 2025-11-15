#include "webcam_capture.h"
#include "torch_inference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// COCO class names (80 classes that RetinaNet can detect)
std::vector<std::string> coco_classes = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

struct Detection
{
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

std::vector<Detection> parseDetections(const torch::Tensor &output, int img_width, int img_height, float conf_threshold = 0.3)
{
    std::vector<Detection> detections;

    std::cout << "Output tensor shape: " << output.sizes() << std::endl;
    std::cout << "Output tensor type: " << output.dtype() << std::endl;

    // If it's a simple tensor (like our previous classification model)
    if (output.dim() == 2 && output.size(1) >= 10)
    {
        // This is a fallback - treat it as our simple detector output
        auto cpu_output = output.cpu();
        auto accessor = cpu_output.accessor<float, 2>();

        // Extract simple detection (center of image)
        Detection det;
        det.bbox = cv::Rect(img_width / 4, img_height / 4, img_width / 2, img_height / 2);
        det.confidence = std::min(1.0f, std::max(0.0f, accessor[0][0])); // First value as confidence
        det.class_id = 0;                                                // Default to "person"
        det.class_name = "detected_object";

        if (det.confidence > conf_threshold)
        {
            detections.push_back(det);
        }
    }

    return detections;
}

void drawDetections(cv::Mat &frame, const std::vector<Detection> &detections)
{
    for (const auto &det : detections)
    {
        // Draw bounding box
        cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);

        // Prepare label text
        std::string label = det.class_name + ": " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        // Get text size for background rectangle
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw label background
        cv::Point label_pos(det.bbox.x, det.bbox.y - text_size.height - 5);
        if (label_pos.y < 0)
            label_pos.y = text_size.height + 5; // Keep label visible

        cv::rectangle(frame,
                      cv::Point(label_pos.x, label_pos.y),
                      cv::Point(label_pos.x + text_size.width, label_pos.y + text_size.height + baseline),
                      cv::Scalar(0, 255, 0),
                      cv::FILLED);

        // Draw label text
        cv::putText(frame, label,
                    cv::Point(label_pos.x, label_pos.y + text_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void drawFPS(cv::Mat &frame, double fps)
{
    std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
}

void drawInferenceTime(cv::Mat &frame, double inference_time_ms)
{
    std::string time_text = "Inference: " + std::to_string(static_cast<int>(inference_time_ms)) + "ms";
    cv::putText(frame, time_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
}

void drawStatus(cv::Mat &frame, bool continuous_detection, bool model_loaded)
{
    std::string status = continuous_detection ? "DETECTING" : "PAUSED";
    cv::Scalar color = continuous_detection ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    if (!model_loaded)
    {
        status = "NO MODEL";
        color = cv::Scalar(0, 0, 255);
    }

    cv::putText(frame, status, cv::Point(10, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
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
    std::string model_path = "../../model.pt";
    bool model_loaded = inference.loadModel(model_path);
    if (!model_loaded)
    {
        std::cerr << "Failed to load model, continuing without inference..." << std::endl;
    }

    // Set device (try CUDA first, fallback to CPU)
    inference.setDevice("cuda");
    inference.printModelInfo();

    std::cout << "\n=== Webcam Object Detection ===" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  'q' or ESC - Quit" << std::endl;
    std::cout << "  'c' - Toggle continuous detection (ON by default)" << std::endl;
    std::cout << "  'i' - Single inference" << std::endl;
    std::cout << "  's' - Save frame" << std::endl;
    std::cout << "  'f' - Show FPS info" << std::endl;
    std::cout << "================================\n"
              << std::endl;

    int frame_count = 0;
    bool continuous_detection = model_loaded; // Start with detection ON if model is loaded
    bool show_fps = true;

    // FPS calculation variables
    auto last_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    double inference_time_ms = 0.0;
    int fps_counter = 0;

    while (true)
    {
        auto frame_start = std::chrono::high_resolution_clock::now();

        cv::Mat frame = webcam.captureFrame();

        if (frame.empty())
        {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }

        // Run continuous detection if enabled
        if (continuous_detection && model_loaded)
        {
            try
            {
                auto inference_start = std::chrono::high_resolution_clock::now();

                torch::Tensor output = inference.predictFromImage(frame);
                std::vector<Detection> detections = parseDetections(output, frame.cols, frame.rows, 0.3);
                drawDetections(frame, detections);

                auto inference_end = std::chrono::high_resolution_clock::now();
                inference_time_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

                // Only print detection count occasionally to avoid spam
                if (!detections.empty() && frame_count % 30 == 0)
                {
                    std::cout << "Detected " << detections.size() << " objects (frame " << frame_count << ")" << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Continuous detection error: " << e.what() << std::endl;
                // Don't disable detection, just continue
            }
        }

        // Calculate FPS
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_diff = std::chrono::duration<double>(current_time - last_time).count();

        if (time_diff >= 1.0) // Update FPS every second
        {
            fps = fps_counter / time_diff;
            fps_counter = 0;
            last_time = current_time;
        }
        fps_counter++;

        // Draw overlays
        if (show_fps)
        {
            drawFPS(frame, fps);
            if (continuous_detection && model_loaded)
            {
                drawInferenceTime(frame, inference_time_ms);
            }
        }
        drawStatus(frame, continuous_detection, model_loaded);

        // Display frame
        cv::imshow("Real-time Object Detection", frame);

        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
        { // 'q' or ESC
            break;
        }
        else if (key == 's')
        {
            std::string filename = "capture_" + std::to_string(frame_count) + ".jpg";
            if (webcam.saveFrame(frame, filename))
            {
                std::cout << "Frame saved as " << filename << std::endl;
            }
        }
        else if (key == 'c')
        {
            if (model_loaded)
            {
                continuous_detection = !continuous_detection;
                std::cout << "Continuous detection: " << (continuous_detection ? "ON" : "OFF") << std::endl;
            }
            else
            {
                std::cout << "Cannot toggle detection - no model loaded!" << std::endl;
            }
        }
        else if (key == 'f')
        {
            show_fps = !show_fps;
            std::cout << "FPS display: " << (show_fps ? "ON" : "OFF") << std::endl;
        }
        else if (key == 'i' && model_loaded)
        {
            try
            {
                std::cout << "Running single inference..." << std::endl;

                auto inference_start = std::chrono::high_resolution_clock::now();
                torch::Tensor output = inference.predictFromImage(frame);
                auto inference_end = std::chrono::high_resolution_clock::now();

                double single_inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

                std::vector<Detection> detections = parseDetections(output, frame.cols, frame.rows, 0.3);

                std::cout << "Inference time: " << single_inference_time << "ms" << std::endl;
                std::cout << "Found " << detections.size() << " objects:" << std::endl;
                for (const auto &det : detections)
                {
                    std::cout << "- " << det.class_name
                              << " (confidence: " << (det.confidence * 100) << "%)"
                              << " at [" << det.bbox.x << ", " << det.bbox.y
                              << ", " << det.bbox.width << ", " << det.bbox.height << "]" << std::endl;
                }

                // Draw detections on a copy of the frame and show it
                cv::Mat detection_frame = frame.clone();
                drawDetections(detection_frame, detections);
                cv::imshow("Single Detection Result", detection_frame);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Inference error: " << e.what() << std::endl;
            }
        }

        frame_count++;
    }

    webcam.stopCapture();
    cv::destroyAllWindows();

    return 0;
}