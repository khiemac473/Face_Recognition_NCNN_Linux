#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>

bool file_exists(const std::string &filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

static inline float intersection_area(const FaceObject& a, const FaceObject& b) {
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
        for (int j = 0; j < (int)picked.size(); j++) {
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

static int detect_retinaface(const cv::Mat& img, std::vector<FaceObject>& objs, const std::string &models_dir) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    if (net.load_param((models_dir + "mnet.25-opt.param").c_str()))
        exit(-1);
    if (net.load_model((models_dir + "mnet.25-opt.bin").c_str()))
        exit(-1);
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

// Lưu kết quả detection (vẽ bbox & landmark) vào file
static void save_detection_result(const cv::Mat& img, const std::vector<FaceObject>& objs, const std::string &out_dir) {
    cv::Mat out_img = img.clone();
    for (size_t i = 0; i < objs.size(); i++) {
        const FaceObject &obj = objs[i];
        cv::rectangle(out_img, obj.rect, cv::Scalar(0, 255, 0));
        for (int j = 0; j < 5; j++) {
            cv::circle(out_img, obj.landmark[j], 2, cv::Scalar(0, 255, 255), -1);
        }
        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = obj.rect.x, y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > out_img.cols) x = out_img.cols - label_size.width;
        cv::rectangle(out_img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(out_img, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite(out_dir + "/detection_result.jpg", out_img);
}

// Alignment: căn chỉnh khuôn mặt dựa trên 5 landmark.
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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [image_filename]\n", argv[0]);
        return -1;
    }
    std::string base_dir = "";
    std::string input_path = "input/" + std::string(argv[1]);
    if (!file_exists(input_path)) {
        input_path = "../input/" + std::string(argv[1]);
        if (file_exists(input_path))
            base_dir = "../";
        else {
            fprintf(stderr, "Không tìm thấy file input trong 'input/' hoặc '../input/'\n");
            return -1;
        }
    }
    std::string models_dir = base_dir + "models/";
    std::string output_dir = base_dir + "output";
    system(("mkdir -p " + output_dir).c_str());
    
    cv::Mat img = cv::imread(input_path, 1);
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s thất bại\n", input_path.c_str());
        return -1;
    }
    
    std::vector<FaceObject> faces;
    detect_retinaface(img, faces, models_dir);
    save_detection_result(img, faces, output_dir);
    
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Mat aligned = align_face(img, faces[i]);
        std::string filename = output_dir + "/face_aligned_" + std::to_string(i) + ".jpg";
        if (!aligned.empty()) {
            if (cv::imwrite(filename, aligned))
                printf("Lưu ảnh alignment: %s\n", filename.c_str());
            else
                fprintf(stderr, "Lỗi lưu ảnh alignment: %s\n", filename.c_str());
        } else {
            fprintf(stderr, "Face alignment thất bại cho khuôn mặt %zu\n", i);
        }
    }
    return 0;
}
