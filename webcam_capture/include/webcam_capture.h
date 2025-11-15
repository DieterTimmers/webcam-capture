#ifndef WEBCAM_CAPTURE_H
#define WEBCAM_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <string>

class WebcamCapture {
public:
    WebcamCapture();
    ~WebcamCapture();
    
    // Initialize camera with device ID (0 = default camera)
    bool initialize(int camera_id = 0);
    
    // Capture a single frame
    cv::Mat captureFrame();
    
    // Start/stop video capture
    bool startCapture();
    void stopCapture();
    
    // Save frame to file
    bool saveFrame(const cv::Mat& frame, const std::string& filename);
    
    // Check if camera is opened
    bool isOpened() const;
    
private:
    cv::VideoCapture cap_;
    bool is_capturing_;
};

#endif // WEBCAM_CAPTURE_H