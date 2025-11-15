#include "webcam_capture.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    WebcamCapture webcam;
    
    // Initialize camera
    if (!webcam.initialize(0)) {
        std::cerr << "Failed to initialize webcam" << std::endl;
        return -1;
    }
    
    // Start capture
    if (!webcam.startCapture()) {
        std::cerr << "Failed to start capture" << std::endl;
        return -1;
    }
    
    std::cout << "Press 'q' to quit, 's' to save frame" << std::endl;
    
    int frame_count = 0;
    while (true) {
        cv::Mat frame = webcam.captureFrame();
        
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        
        // Display frame
        cv::imshow("Webcam Capture", frame);
        
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 's') {
            std::string filename = "capture_" + std::to_string(frame_count++) + ".jpg";
            if (webcam.saveFrame(frame, filename)) {
                std::cout << "Frame saved as " << filename << std::endl;
            }
        }
    }
    
    webcam.stopCapture();
    cv::destroyAllWindows();
    
    return 0;
}