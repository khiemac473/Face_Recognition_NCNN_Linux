#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

// Hàm đọc embedding từ file
std::vector<float> loadEmbedding(const std::string &filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Không mở được file: " + filename);
    }
    
    std::string line;
    // Đọc dòng header
    std::getline(ifs, line);
    // Đọc dòng chứa vector embedding
    std::getline(ifs, line);
    
    std::istringstream iss(line);
    std::vector<float> embedding;
    float val;
    while (iss >> val) {
        embedding.push_back(val);
    }
    return embedding;
}

// Hàm tính cosine similarity giữa 2 vector
float cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size() || v1.empty()) {
        throw std::runtime_error("Các vector phải có kích thước bằng nhau và không rỗng.");
    }
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    if (norm1 == 0 || norm2 == 0) return 0.0f;
    return dot / (norm1 * norm2);
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [embedding_file1] [embedding_file2]" << std::endl;
        return -1;
    }
    
    try {
        std::vector<float> emb1 = loadEmbedding(argv[1]);
        std::vector<float> emb2 = loadEmbedding(argv[2]);
        float cosine = cosineSimilarity(emb1, emb2);
        std::cout << "Cosine similarity: " << cosine << std::endl;
        
        // Đặt ngưỡng
        float threshold = 0.5f;
        if (cosine >= threshold)
            std::cout << "Compare: SAME PERSON" << std::endl;
        else
            std::cout << "Compare: DIFFERENT PERSON" << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Lỗi: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

