#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>
#include <vector>

// Runtime knobs. All values can be overridden from CLI flags.
struct AppConfig {
    // Path to YuNet ONNX model file.
    std::string model_path = "face_detection_yunet_2023mar.onnx";
    // Which webcam to open (0 = default camera).
    int camera_index = 0;
    // Minimum confidence score for a detected face.
    float score_threshold = 0.8f;
    // Non-maximum suppression threshold for overlapping detections.
    float nms_threshold = 0.3f;
    // Candidate boxes before NMS. Keep high unless performance issues appear.
    int top_k = 5000;
    // Pixelation strength. Higher => larger blocks => stronger anonymization.
    int pixel_block = 28;
    // Expand face box on all sides. Helps hide face edges better.
    float face_padding = 0.5f;
    // Keep using previous face boxes for a few frames if detection drops briefly.
    int hold_frames = 20;
};

// Ensure rectangle is inside frame boundaries.
static cv::Rect clamp_rect(const cv::Rect& r, int width, int height) {
    int x1 = std::max(0, r.x);
    int y1 = std::max(0, r.y);
    int x2 = std::min(width, r.x + r.width);
    int y2 = std::min(height, r.y + r.height);
    return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

// Expand a detected face rectangle for safer privacy masking.
static cv::Rect expand_rect(const cv::Rect& r, float pad_ratio, int width, int height) {
    int pad_w = static_cast<int>(r.width * pad_ratio);
    int pad_h = static_cast<int>(r.height * pad_ratio);
    cv::Rect expanded(r.x - pad_w, r.y - pad_h, r.width + 2 * pad_w, r.height + 2 * pad_h);
    return clamp_rect(expanded, width, height);
}

// Pixelate region by downscaling and scaling back with nearest-neighbor.
static cv::Mat pixelate_roi(const cv::Mat& roi, int block_size) {
    if (roi.empty()) {
        return roi;
    }
    block_size = std::max(2, block_size);
    int small_w = std::max(1, roi.cols / block_size);
    int small_h = std::max(1, roi.rows / block_size);

    cv::Mat small;
    cv::resize(roi, small, cv::Size(small_w, small_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat pixelated;
    cv::resize(small, pixelated, roi.size(), 0, 0, cv::INTER_NEAREST);
    return pixelated;
}

// Minimal CLI parser for app options.
static AppConfig parse_args(int argc, char** argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(1);
            }
        };

        if (key == "--model") {
            need_value(key);
            cfg.model_path = argv[++i];
        } else if (key == "--camera") {
            need_value(key);
            cfg.camera_index = std::stoi(argv[++i]);
        } else if (key == "--score-threshold") {
            need_value(key);
            cfg.score_threshold = std::stof(argv[++i]);
        } else if (key == "--nms-threshold") {
            need_value(key);
            cfg.nms_threshold = std::stof(argv[++i]);
        } else if (key == "--top-k") {
            need_value(key);
            cfg.top_k = std::stoi(argv[++i]);
        } else if (key == "--pixel-block") {
            need_value(key);
            cfg.pixel_block = std::stoi(argv[++i]);
        } else if (key == "--face-padding") {
            need_value(key);
            cfg.face_padding = std::stof(argv[++i]);
        } else if (key == "--hold-frames") {
            need_value(key);
            cfg.hold_frames = std::stoi(argv[++i]);
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: face_pixelate_cpp [options]\n"
                      << "  --model <path>            YuNet model path\n"
                      << "  --camera <index>          Camera index (default 0)\n"
                      << "  --score-threshold <f>     Detector score threshold\n"
                      << "  --nms-threshold <f>       NMS threshold\n"
                      << "  --top-k <int>             Top-K before NMS\n"
                      << "  --pixel-block <int>       Pixelation strength\n"
                      << "  --face-padding <f>        Extra mask padding ratio\n"
                      << "  --hold-frames <int>       Frames to keep last boxes\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << key << std::endl;
            std::exit(1);
        }
    }

    // Keep values in safe ranges.
    cfg.pixel_block = std::max(2, cfg.pixel_block);
    cfg.hold_frames = std::max(0, cfg.hold_frames);
    cfg.face_padding = std::max(0.0f, cfg.face_padding);
    return cfg;
}

int main(int argc, char** argv) {
    AppConfig cfg = parse_args(argc, argv);

    // 1) Open camera.
    cv::VideoCapture cap(cfg.camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera index " << cfg.camera_index << std::endl;
        return 1;
    }

    // Read one frame first to initialize detector with real frame size.
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to read initial frame from camera." << std::endl;
        return 1;
    }

    // 2) Create YuNet neural face detector.
    auto detector = cv::FaceDetectorYN::create(
        cfg.model_path,
        "",
        frame.size(),
        cfg.score_threshold,
        cfg.nms_threshold,
        cfg.top_k);

    if (detector.empty()) {
        std::cerr << "Failed to create YuNet detector. Check model path: " << cfg.model_path << std::endl;
        return 1;
    }

    std::vector<cv::Rect> last_boxes;
    int missed_frames = 0;

    std::cout << "Press q or ESC to quit." << std::endl;
    // 3) Main processing loop.
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // Detect faces on current frame.
        detector->setInputSize(frame.size());
        cv::Mat faces;
        detector->detect(frame, faces);

        std::vector<cv::Rect> current_boxes;
        if (!faces.empty()) {
            // YuNet returns rows of floats. First 4 values are x, y, w, h.
            for (int i = 0; i < faces.rows; ++i) {
                const float* row = faces.ptr<float>(i);
                cv::Rect box(
                    static_cast<int>(row[0]),
                    static_cast<int>(row[1]),
                    static_cast<int>(row[2]),
                    static_cast<int>(row[3]));
                box = expand_rect(box, cfg.face_padding, frame.cols, frame.rows);
                if (box.width > 0 && box.height > 0) {
                    current_boxes.push_back(box);
                }
            }
            if (!current_boxes.empty()) {
                // Fresh detection: remember boxes and reset dropout counter.
                last_boxes = current_boxes;
                missed_frames = 0;
            }
        } else if (!last_boxes.empty() && missed_frames < cfg.hold_frames) {
            // Detection dropped this frame: keep previous boxes temporarily.
            current_boxes = last_boxes;
            missed_frames++;
        } else {
            // Too many misses: stop using stale boxes.
            last_boxes.clear();
        }

        // 4) Pixelate detected regions + draw debug rectangle.
        for (const auto& box : current_boxes) {
            cv::Mat roi = frame(box);
            cv::Mat pix = pixelate_roi(roi, cfg.pixel_block);
            pix.copyTo(frame(box));
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }

        // 5) Show output and handle quit key.
        cv::imshow("YuNet Face Pixelate (C++)", frame);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
