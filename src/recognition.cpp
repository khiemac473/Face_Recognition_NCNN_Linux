#include "net.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Structure representing a detected face.
struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

// Utility functions for face detection.
static inline float intersection_area(const FaceObject &a, const FaceObject &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void sort_faceobjects(std::vector<FaceObject>& objs) {
    std::sort(objs.begin(), objs.end(), [](const FaceObject &a, const FaceObject &b) {
        return a.prob > b.prob;
    });
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& objs, std::vector<int>& picked, float threshold) {
    picked.clear();
    int n = objs.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objs[i].rect.area();
    }
    for (int i = 0; i < n; i++) {
        const FaceObject& a = objs[i];
        bool keep = true;
        for (int j = 0; j < static_cast<int>(picked.size()); j++) {
            const FaceObject& b = objs[picked[j]];
            float inter = intersection_area(a, b);
            float uni = areas[i] + areas[picked[j]] - inter;
            if (inter / uni > threshold) {
                keep = false;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales) {
    int num_ratio = ratios.w, num_scale = scales.w;
    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);
    const float cx = base_size * 0.5f, cy = base_size * 0.5f;
    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];
        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);
        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];
            float rs_w = r_w * scale, rs_h = r_h * scale;
            float* anchor = anchors.row(i * num_scale + j);
            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }
    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride,
                               const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob,
                               const ncnn::Mat& landmark_blob, float prob_threshold,
                               std::vector<FaceObject>& objs) {
    int w = score_blob.w, h = score_blob.h;
    const int num_anchors = anchors.h;
    for (int q = 0; q < num_anchors; q++) {
        const float* anchor = anchors.row(q);
        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);
        float anchor_y = anchor[1];
        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];
        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float prob = score[index];
                if (prob >= prob_threshold) {
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;
                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;
                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);
                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;
                    FaceObject obj;
                    obj.rect = cv::Rect_<float>(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
                    obj.landmark[0].x = cx + (anchor_w + 1) * landmark.channel(0)[index];
                    obj.landmark[0].y = cy + (anchor_h + 1) * landmark.channel(1)[index];
                    obj.landmark[1].x = cx + (anchor_w + 1) * landmark.channel(2)[index];
                    obj.landmark[1].y = cy + (anchor_h + 1) * landmark.channel(3)[index];
                    obj.landmark[2].x = cx + (anchor_w + 1) * landmark.channel(4)[index];
                    obj.landmark[2].y = cy + (anchor_h + 1) * landmark.channel(5)[index];
                    obj.landmark[3].x = cx + (anchor_w + 1) * landmark.channel(6)[index];
                    obj.landmark[3].y = cy + (anchor_h + 1) * landmark.channel(7)[index];
                    obj.landmark[4].x = cx + (anchor_w + 1) * landmark.channel(8)[index];
                    obj.landmark[4].y = cy + (anchor_h + 1) * landmark.channel(9)[index];
                    obj.prob = prob;
                    objs.push_back(obj);
                }
                anchor_x += feat_stride;
            }
            anchor_y += feat_stride;
        }
    }
}

static int detect_retinaface(const cv::Mat& img, std::vector<FaceObject>& objs, ncnn::Net &net) {
    const float prob_threshold = 0.8f, nms_threshold = 0.4f;
    int img_w = img.cols, img_h = img.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);
    
    std::vector<FaceObject> proposals;
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);
        int base_size = 16, feat_stride = 32;
        ncnn::Mat ratios(1); ratios[0] = 1.f;
        ncnn::Mat scales(2); scales[0] = 32.f; scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
        std::vector<FaceObject> objs32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, objs32);
        proposals.insert(proposals.end(), objs32.begin(), objs32.end());
    }
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);
        int base_size = 16, feat_stride = 16;
        ncnn::Mat ratios(1); ratios[0] = 1.f;
        ncnn::Mat scales(2); scales[0] = 8.f; scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
        std::vector<FaceObject> objs16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, objs16);
        proposals.insert(proposals.end(), objs16.begin(), objs16.end());
    }
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);
        int base_size = 16, feat_stride = 8;
        ncnn::Mat ratios(1); ratios[0] = 1.f;
        ncnn::Mat scales(2); scales[0] = 2.f; scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
        std::vector<FaceObject> objs8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, objs8);
        proposals.insert(proposals.end(), objs8.begin(), objs8.end());
    }
    sort_faceobjects(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objs.resize(count);
    for (int i = 0; i < count; i++) {
        objs[i] = proposals[picked[i]];
        float x0 = std::max(std::min(objs[i].rect.x,  (float)img_w - 1), 0.f);
        float y0 = std::max(std::min(objs[i].rect.y,  (float)img_h - 1), 0.f);
        float x1 = std::max(std::min(objs[i].rect.x + objs[i].rect.width,  (float)img_w - 1), 0.f);
        float y1 = std::max(std::min(objs[i].rect.y + objs[i].rect.height, (float)img_h - 1), 0.f);
        objs[i].rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
    }
    return 0;
}

// Align face based on 5 landmarks.
cv::Mat align_face(const cv::Mat &img, const FaceObject &obj) {
    std::vector<cv::Point2f> dst = {
        cv::Point2f(38.2946f, 51.6963f),
        cv::Point2f(73.5318f, 51.5014f),
        cv::Point2f(56.0252f, 71.7366f),
        cv::Point2f(41.5493f, 92.3655f),
        cv::Point2f(70.7299f, 92.2041f)
    };
    std::vector<cv::Point2f> src;
    for (int i = 0; i < 5; i++) {
        src.push_back(obj.landmark[i]);
    }
    cv::Mat trans = cv::estimateAffinePartial2D(src, dst);
    if (trans.empty()) {
        cv::Rect roi(cvRound(obj.rect.x), cvRound(obj.rect.y),
                     cvRound(obj.rect.width), cvRound(obj.rect.height));
        roi &= cv::Rect(0, 0, img.cols, img.rows);
        cv::Mat cropped = img(roi);
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(112, 112));
        return resized;
    }
    cv::Mat aligned;
    cv::warpAffine(img, aligned, trans, cv::Size(112, 112));
    return aligned;
}

// Embedding helper functions.
std::vector<float> loadEmbedding(const std::string &filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: Unable to open embedding file: " << filename << std::endl;
        return {};
    }
    std::string header;
    std::getline(ifs, header);
    std::string line;
    std::getline(ifs, line);
    std::istringstream iss(line);
    std::vector<float> embedding;
    float val;
    while (iss >> val) {
        embedding.push_back(val);
    }
    return embedding;
}

float cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size() || v1.empty())
        return 0.0f;
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    if (norm1 == 0 || norm2 == 0) return 0.0f;
    return dot / (norm1 * norm2);
}

std::vector<float> compute_embedding(const cv::Mat &aligned, ncnn::Net &arcface) {
    ncnn::Mat in = ncnn::Mat::from_pixels(aligned.data, ncnn::Mat::PIXEL_BGR2RGB, aligned.cols, aligned.rows);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = arcface.create_extractor();
    ex.input("input.1", in);
    ncnn::Mat feat;
    if (ex.extract("683", feat) != 0) {
        std::cerr << "Error: Failed to extract embedding from ArcFace." << std::endl;
        return {};
    }
    std::vector<float> embedding(feat.w);
    for (int i = 0; i < feat.w; i++) {
        embedding[i] = feat[i];
    }
    return embedding;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [embedding_file] (optional: [camera_index])" << std::endl;
        return -1;
    }
    std::string ref_embedding_file = argv[1];
    int camera_index = 0;
    if (argc >= 3) {
        camera_index = std::stoi(argv[2]);
    }

    // Load reference embedding from provided file.
    std::vector<float> refEmbedding = loadEmbedding(ref_embedding_file);
    if (refEmbedding.empty()) {
        std::cerr << "Error: Failed to load reference embedding from file " << ref_embedding_file << std::endl;
        return -1;
    }

    // Load RetinaFace model.
    ncnn::Net retinaface;
    retinaface.opt.use_vulkan_compute = true;
    std::string models_dir = "../models/";
    if (retinaface.load_param((models_dir + "mnet.25-opt.param").c_str()) != 0) {
        std::cerr << "Error: Failed to load RetinaFace param." << std::endl;
        return -1;
    }
    if (retinaface.load_model((models_dir + "mnet.25-opt.bin").c_str()) != 0) {
        std::cerr << "Error: Failed to load RetinaFace model." << std::endl;
        return -1;
    }

    // Load ArcFace model.
    ncnn::Net arcface;
    arcface.opt.use_vulkan_compute = true;
    if (arcface.load_param((models_dir + "arcface.param").c_str()) != 0) {
        std::cerr << "Error: Failed to load ArcFace param." << std::endl;
        return -1;
    }
    if (arcface.load_model((models_dir + "arcface.bin").c_str()) != 0) {
        std::cerr << "Error: Failed to load ArcFace model." << std::endl;
        return -1;
    }

    // Open the camera.
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera." << std::endl;
        return -1;
    }
    std::cout << "Press 'q' or ESC to exit." << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // Detect faces in the current frame.
        std::vector<FaceObject> faces;
        detect_retinaface(frame, faces, retinaface);

        // For each detected face, align, extract embedding and compare with reference.
        for (size_t i = 0; i < faces.size(); i++) {
            cv::Mat aligned = align_face(frame, faces[i]);
            if (aligned.empty())
                continue;
            std::vector<float> faceEmbedding = compute_embedding(aligned, arcface);
            if (faceEmbedding.empty())
                continue;
            float similarity = cosineSimilarity(refEmbedding, faceEmbedding);
            std::string label = (similarity >= 0.5f) ? "SAME PERSON" : "DIFFERENT PERSON";
            std::string text = label + " (" + std::to_string(similarity) + ")";
            cv::rectangle(frame, faces[i].rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, text, cv::Point(faces[i].rect.x, faces[i].rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow("Real-time Face Recognition", frame);
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
