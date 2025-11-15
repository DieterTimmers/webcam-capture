#include "webcam_capture.h"
#include <iostream>

WebcamCapture::WebcamCapture() : is_capturing_(false) {
}

WebcamCapture::~WebcamCapture() {
    stopCapture();
}

bool WebcamCapture::initialize(int camera_id) {
    cap_.open(camera_id);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open camera " << camera_id << std::endl;
        return false;
    }
    
    // Set some default properties
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    std::cout << "Camera initialized successfully" << std::endl;
    return true;
}

cv::Mat WebcamCapture::captureFrame() {
    cv::Mat frame;
    if (cap_.isOpened()) {
        cap_ >> frame;
    }
    return frame;
}

bool WebcamCapture::startCapture() {
    if (!cap_.isOpened()) {
        std::cerr << "Error: Camera not initialized" << std::endl;
        return false;
    }
    is_capturing_ = true;
    return true;
}

void WebcamCapture::stopCapture() {
    is_capturing_ = false;
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool WebcamCapture::saveFrame(const cv::Mat& frame, const std::string& filename) {
    if (frame.empty()) {
        std::cerr << "Error: Frame is empty" << std::endl;
        return false;
    }
    
    return cv::imwrite(filename, frame);
}

bool WebcamCapture::isOpened() const {
    return cap_.isOpened();
}