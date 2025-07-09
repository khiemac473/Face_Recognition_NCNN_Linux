#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

// Hàm kiểm tra sự tồn tại của file
bool file_exists(const std::string &filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

// Hàm lấy tên file gốc
std::string getBaseName(const std::string &filepath) {
    size_t pos = filepath.find_last_of("/\\");
    std::string filename = (pos == std::string::npos) ? filepath : filepath.substr(pos + 1);
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        return filename.substr(0, dot_pos);
    }
    return filename;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [aligned_face_image]" << std::endl;
        return -1;
    }
    
    std::string aligned_img_path = argv[1];
    if (!file_exists(aligned_img_path)) {
        std::cerr << "Khong thay file anh: " << aligned_img_path << std::endl;
        return -1;
    }
    
    // Đọc ảnh đã alignment (112×112)
    cv::Mat aligned = cv::imread(aligned_img_path, cv::IMREAD_COLOR);
    if (aligned.empty()) {
        std::cerr << "Loi doc anh: " << aligned_img_path << std::endl;
        return -1;
    }
    
    // Đường dẫn tới model 
    std::string models_dir = "../models/";
    
    ncnn::Net arcface;
    arcface.opt.use_vulkan_compute = true;
    if (arcface.load_param((models_dir + "arcface.param").c_str()) != 0) {
        std::cerr << "Loi tai model: " << models_dir + "arcface.param" << std::endl;
        return -1;
    }
    if (arcface.load_model((models_dir + "arcface.bin").c_str()) != 0) {
        std::cerr << "Loi tai model: " << models_dir + "arcface.bin" << std::endl;
        return -1;
    }
    
    // Tiền xử lý: chuyển ảnh từ BGR sang RGB và normalize với mean=127.5, scale=1/128
    ncnn::Mat in = ncnn::Mat::from_pixels(aligned.data, ncnn::Mat::PIXEL_BGR2RGB, aligned.cols, aligned.rows);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    ncnn::Extractor ex = arcface.create_extractor();
    ex.input("input.1", in); // Tên input layer theo file param của bạn
    
    ncnn::Mat feat;
    // Sử dụng layer "683" để trích xuất embedding
    if (ex.extract("683", feat) != 0) {
        std::cerr << "Loi extract embedding tu layer \"683\"" << std::endl;
        return -1;
    }
    
    if (feat.w <= 0) {
        std::cerr << "Embedding vector trong" << std::endl;
        return -1;
    }
    
    // Tạo thư mục output nếu chưa tồn tại
    system("mkdir -p ../output/embedding");
    
    // Tạo tên file embedding dựa trên tên ảnh gốc
    std::string baseName = getBaseName(aligned_img_path);
    std::string output_file = "../output/embedding/" + baseName + "_embedding.txt";
    
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Cant open file write embedding: " << output_file << std::endl;
        return -1;
    }
    
    ofs << "Embedding vector for " << aligned_img_path << ":\n";
    for (int i = 0; i < feat.w; i++) {
        ofs << feat[i] << " ";
    }
    ofs << "\n";
    ofs.close();
    
    std::cout << "Done embedding: " << output_file << std::endl;
    
    return 0;
}

