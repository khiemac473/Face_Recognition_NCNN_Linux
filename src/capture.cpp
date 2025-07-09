#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
	
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cant open camera!" << std::endl;
        return -1;
    }

    std::string save_dir = "../input/capture/";
    if (!fs::exists(save_dir)) {
        fs::create_directories(save_dir);
    }

    cv::Mat frame;
    int img_counter = 0;

    std::cout << "'s' to save, 'q' to exit" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("Camera", frame);
        char key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 's') { 
            std::string img_name = save_dir + "capture_" + std::to_string(img_counter) + ".jpg";
            cv::imwrite(img_name, frame);
            std::cout << "SaveImg: " << img_name << std::endl;
            img_counter++;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

