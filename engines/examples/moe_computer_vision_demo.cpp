// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../../common/src/result.hpp"
#include "../src/mixture_experts/moe_config.hpp"
#include "../src/mixture_experts/moe_engine.hpp"

using namespace engines::mixture_experts;
using namespace inference_lab::common;

// ================================================================================================
// IMAGE REPRESENTATION AND FEATURE EXTRACTION
// ================================================================================================

/**
 * @brief Simplified image representation for demonstration
 *
 * In a real application, this would handle:
 * - Raw pixel data (RGB/grayscale)
 * - Pre-trained feature extractors (ResNet, VGG, etc.)
 * - Data augmentation and preprocessing
 * - GPU memory management
 */
struct ImageData {
    std::vector<float> pixels;  // Flattened pixel values [0,1]
    std::size_t width;
    std::size_t height;
    std::size_t channels;

    static constexpr std::size_t DEFAULT_SIZE = 64;  // 64x64 for demo
    static constexpr std::size_t CHANNELS = 3;       // RGB
    static constexpr std::size_t FEATURE_DIM = 512;  // CNN feature dimension

    ImageData(std::size_t w = DEFAULT_SIZE, std::size_t h = DEFAULT_SIZE, std::size_t c = CHANNELS)
        : width(w), height(h), channels(c) {
        pixels.resize(w * h * c, 0.0f);
    }

    /**
     * @brief Generate synthetic image data for demonstration
     */
    static ImageData generate_synthetic(std::mt19937& gen, const std::string& image_type) {
        ImageData image;
        std::uniform_real_distribution<float> pixel_dist(0.0f, 1.0f);

        // Generate different patterns based on image type to simulate different visual tasks
        for (std::size_t i = 0; i < image.pixels.size(); ++i) {
            if (image_type == "object") {
                // Objects: More structured patterns with edges
                std::size_t x = (i / image.channels) % image.width;
                std::size_t y = (i / image.channels) / image.width;
                image.pixels[i] = std::sin(x * 0.1f) * std::cos(y * 0.1f) * 0.5f + 0.5f;
            } else if (image_type == "scene") {
                // Scenes: Smooth gradients and textures
                image.pixels[i] = pixel_dist(gen) * 0.8f + 0.1f;
            } else if (image_type == "face") {
                // Faces: Circular patterns with high contrast
                std::size_t x = (i / image.channels) % image.width;
                std::size_t y = (i / image.channels) / image.width;
                float center_x = image.width / 2.0f;
                float center_y = image.height / 2.0f;
                float dist =
                    std::sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y));
                image.pixels[i] = std::exp(-dist * 0.05f);
            } else {
                // Default: Random noise
                image.pixels[i] = pixel_dist(gen);
            }
        }

        return image;
    }

    /**
     * @brief Extract CNN-style features (simplified demonstration)
     *
     * In a real implementation, this would use:
     * - Convolutional layers with learned kernels
     * - Pooling operations for translation invariance
     * - Pre-trained feature extractors (ResNet features)
     * - GPU acceleration with optimized convolution libraries
     */
    std::vector<float> extract_cnn_features() const {
        std::vector<float> features(FEATURE_DIM, 0.0f);

        // Simulate convolutional feature extraction
        // Conv Layer 1: Edge detection (simplified)
        for (std::size_t i = 1; i < height - 1; ++i) {
            for (std::size_t j = 1; j < width - 1; ++j) {
                for (std::size_t c = 0; c < channels; ++c) {
                    std::size_t idx = i * width * channels + j * channels + c;

                    // Simple edge detection kernel
                    float edge_response =
                        -pixels[idx - width * channels - channels] +  // Top-left
                        -pixels[idx - width * channels] +             // Top
                        -pixels[idx - width * channels + channels] +  // Top-right
                        -pixels[idx - channels] +                     // Left
                        8 * pixels[idx] +                             // Center
                        -pixels[idx + channels] +                     // Right
                        -pixels[idx + width * channels - channels] +  // Bottom-left
                        -pixels[idx + width * channels] +             // Bottom
                        -pixels[idx + width * channels + channels];   // Bottom-right

                    // ReLU activation
                    edge_response = std::max(0.0f, edge_response);

                    // Add to feature vector (simplified pooling)
                    std::size_t feature_idx = (i * width + j) % FEATURE_DIM;
                    features[feature_idx] += edge_response / (channels * 255.0f);
                }
            }
        }

        // Simulate additional convolutional layers with different feature extractors
        // Conv Layer 2: Texture features
        for (std::size_t i = 0; i < FEATURE_DIM / 2; ++i) {
            float texture_feature = 0.0f;
            for (std::size_t j = i; j < pixels.size(); j += FEATURE_DIM) {
                texture_feature += pixels[j % pixels.size()] * std::sin(j * 0.01f);
            }
            features[i + FEATURE_DIM / 2] = std::abs(texture_feature) / 100.0f;
        }

        // Global pooling and normalization
        float feature_sum = 0.0f;
        for (float f : features) {
            feature_sum += f * f;
        }
        if (feature_sum > 0.0f) {
            float norm = std::sqrt(feature_sum);
            for (float& f : features) {
                f /= norm;
            }
        }

        return features;
    }
};

// ================================================================================================
// COMPUTER VISION TASKS AND LABELS
// ================================================================================================

/**
 * @brief Different computer vision tasks that experts specialize in
 */
enum class VisionTask : std::uint8_t {
    OBJECT_DETECTION = 0,      // Detecting and classifying objects
    SCENE_CLASSIFICATION = 1,  // Classifying scenes and environments
    FACIAL_RECOGNITION = 2,    // Face detection and recognition
    UNKNOWN = 255
};

/**
 * @brief Classification results for different vision tasks
 */
enum class VisionLabel : std::uint8_t {
    // Object detection
    PERSON = 0,
    CAR = 1,
    BUILDING = 2,
    // Scene classification
    INDOOR = 0,
    OUTDOOR = 1,
    URBAN = 2,
    // Facial recognition
    FACE_DETECTED = 0,
    NO_FACE = 1,
    MULTIPLE_FACES = 2
};

/**
 * @brief Image sample with ground truth for demonstration
 */
struct ImageSample {
    ImageData image;
    VisionTask task;
    VisionLabel label;
    std::string description;

    ImageSample(ImageData img, VisionTask t, VisionLabel l, std::string desc)
        : image(std::move(img)), task(t), label(l), description(std::move(desc)) {}
};

// ================================================================================================
// COMPUTER VISION EXPERT NETWORKS
// ================================================================================================

/**
 * @brief Specialized CNN-style expert for different computer vision tasks
 *
 * In a real implementation, this would interface with:
 * - PyTorch/TensorFlow pre-trained models
 * - ONNX Runtime for cross-platform inference
 * - TensorRT for GPU acceleration
 * - OpenCV for image preprocessing
 */
class ComputerVisionExpert {
  private:
    VisionTask specialization_;
    std::size_t expert_id_;

    // Simplified CNN architecture simulation
    struct ConvLayer {
        std::vector<std::vector<float>> filters;  // Convolution filters
        std::vector<float> biases;
        std::size_t input_channels;
        std::size_t output_channels;
        std::size_t kernel_size;
    };

    std::vector<ConvLayer> conv_layers_;
    std::vector<std::vector<float>> fc_weights_;  // Fully connected layers
    std::vector<float> fc_biases_;

  public:
    explicit ComputerVisionExpert(VisionTask specialization, std::size_t expert_id)
        : specialization_(specialization), expert_id_(expert_id) {
        initialize_cnn_architecture();
    }

    /**
     * @brief Initialize CNN architecture specialized for the task
     */
    void initialize_cnn_architecture() {
        // Task-specific architecture initialization
        std::mt19937 gen(42 + static_cast<std::size_t>(specialization_) * 1000 + expert_id_);
        std::normal_distribution<float> weight_dist(0.0f, 0.1f);

        // Different architectures for different tasks
        switch (specialization_) {
            case VisionTask::OBJECT_DETECTION:
                initialize_object_detection_net(gen, weight_dist);
                break;
            case VisionTask::SCENE_CLASSIFICATION:
                initialize_scene_classification_net(gen, weight_dist);
                break;
            case VisionTask::FACIAL_RECOGNITION:
                initialize_facial_recognition_net(gen, weight_dist);
                break;
            default:
                initialize_generic_net(gen, weight_dist);
                break;
        }

        // Final fully connected layers (shared structure)
        constexpr std::size_t FC_INPUT_DIM = 256;  // From conv layers
        constexpr std::size_t FC_HIDDEN_DIM = 128;
        constexpr std::size_t FC_OUTPUT_DIM = 3;  // 3 classes per task

        // FC Layer 1
        fc_weights_.push_back(std::vector<float>(FC_INPUT_DIM * FC_HIDDEN_DIM));
        for (auto& weight : fc_weights_.back()) {
            weight = weight_dist(gen);
        }
        fc_biases_.resize(FC_HIDDEN_DIM);
        for (auto& bias : fc_biases_) {
            bias = weight_dist(gen);
        }

        // FC Layer 2 (Output)
        fc_weights_.push_back(std::vector<float>(FC_HIDDEN_DIM * FC_OUTPUT_DIM));
        for (auto& weight : fc_weights_.back()) {
            weight = weight_dist(gen);
        }
        std::vector<float> output_biases(FC_OUTPUT_DIM);
        for (auto& bias : output_biases) {
            bias = weight_dist(gen);
        }
        fc_biases_.insert(fc_biases_.end(), output_biases.begin(), output_biases.end());
    }

    /**
     * @brief Initialize object detection specialized architecture
     * Focus on edge detection and spatial features
     */
    void initialize_object_detection_net(std::mt19937& gen, std::normal_distribution<float>& dist) {
        // Layer 1: Edge detection filters
        ConvLayer layer1;
        layer1.input_channels = 3;
        layer1.output_channels = 32;
        layer1.kernel_size = 3;

        for (std::size_t i = 0; i < layer1.output_channels; ++i) {
            layer1.filters.emplace_back(layer1.input_channels * layer1.kernel_size *
                                        layer1.kernel_size);
            for (auto& weight : layer1.filters.back()) {
                weight = dist(gen);
            }
            layer1.biases.push_back(dist(gen));
        }
        conv_layers_.push_back(std::move(layer1));
    }

    /**
     * @brief Initialize scene classification specialized architecture
     * Focus on global texture and color features
     */
    void initialize_scene_classification_net(std::mt19937& gen,
                                             std::normal_distribution<float>& dist) {
        // Layer 1: Texture analysis filters
        ConvLayer layer1;
        layer1.input_channels = 3;
        layer1.output_channels = 16;  // Fewer filters for global features
        layer1.kernel_size = 5;       // Larger kernel for global patterns

        for (std::size_t i = 0; i < layer1.output_channels; ++i) {
            layer1.filters.emplace_back(layer1.input_channels * layer1.kernel_size *
                                        layer1.kernel_size);
            for (auto& weight : layer1.filters.back()) {
                weight = dist(gen) * 0.8f;  // Smaller weights for global features
            }
            layer1.biases.push_back(dist(gen));
        }
        conv_layers_.push_back(std::move(layer1));
    }

    /**
     * @brief Initialize facial recognition specialized architecture
     * Focus on high-frequency patterns and symmetry
     */
    void initialize_facial_recognition_net(std::mt19937& gen,
                                           std::normal_distribution<float>& dist) {
        // Layer 1: High-frequency pattern detection
        ConvLayer layer1;
        layer1.input_channels = 3;
        layer1.output_channels = 64;  // More filters for detailed features
        layer1.kernel_size = 3;

        for (std::size_t i = 0; i < layer1.output_channels; ++i) {
            layer1.filters.emplace_back(layer1.input_channels * layer1.kernel_size *
                                        layer1.kernel_size);
            for (auto& weight : layer1.filters.back()) {
                weight = dist(gen) * 1.2f;  // Larger weights for fine details
            }
            layer1.biases.push_back(dist(gen));
        }
        conv_layers_.push_back(std::move(layer1));
    }

    /**
     * @brief Generic CNN architecture for unknown tasks
     */
    void initialize_generic_net(std::mt19937& gen, std::normal_distribution<float>& dist) {
        ConvLayer layer1;
        layer1.input_channels = 3;
        layer1.output_channels = 24;
        layer1.kernel_size = 4;

        for (std::size_t i = 0; i < layer1.output_channels; ++i) {
            layer1.filters.emplace_back(layer1.input_channels * layer1.kernel_size *
                                        layer1.kernel_size);
            for (auto& weight : layer1.filters.back()) {
                weight = dist(gen);
            }
            layer1.biases.push_back(dist(gen));
        }
        conv_layers_.push_back(std::move(layer1));
    }

    /**
     * @brief Forward pass through the CNN expert
     */
    std::vector<float> forward(const std::vector<float>& input_features) {
        // For this demo, we'll work with pre-extracted features
        // In a real implementation, this would process raw image pixels through conv layers

        // Simulate convolutional processing (simplified)
        std::vector<float> conv_output = simulate_conv_forward(input_features);

        // Fully connected layers
        std::vector<float> fc1_output = apply_fully_connected(conv_output, 0);

        // Apply ReLU activation
        for (auto& val : fc1_output) {
            val = std::max(0.0f, val);
        }

        // Final output layer
        std::vector<float> output = apply_fully_connected(fc1_output, 1);

        // Apply softmax
        apply_softmax(output);

        return output;
    }

  private:
    /**
     * @brief Simulate convolutional forward pass
     */
    std::vector<float> simulate_conv_forward(const std::vector<float>& input) {
        std::vector<float> output(256, 0.0f);  // Fixed size for demo

        // Simulate convolution with learned filters
        for (std::size_t i = 0; i < output.size(); ++i) {
            for (std::size_t j = 0; j < std::min(input.size(), static_cast<std::size_t>(32)); ++j) {
                // Simulate filter response based on specialization
                float weight = specialization_ == VisionTask::OBJECT_DETECTION       ? 1.2f
                               : specialization_ == VisionTask::SCENE_CLASSIFICATION ? 0.8f
                               : specialization_ == VisionTask::FACIAL_RECOGNITION   ? 1.5f
                                                                                     : 1.0f;

                output[i] += input[j] * weight * std::sin((i + j) * 0.1f);
            }
            // Apply bias and activation
            output[i] = std::max(0.0f, output[i] + static_cast<float>(i % 100) * 0.001f);
        }

        return output;
    }

    /**
     * @brief Apply fully connected layer
     */
    std::vector<float> apply_fully_connected(const std::vector<float>& input,
                                             std::size_t layer_idx) {
        if (layer_idx >= fc_weights_.size()) {
            return input;
        }

        std::size_t output_size = (layer_idx == 0) ? 128 : 3;  // Hidden size or output size
        std::size_t input_size = input.size();

        std::vector<float> output(output_size, 0.0f);

        for (std::size_t i = 0; i < output_size; ++i) {
            // Add bias
            output[i] = (layer_idx == 0) ? fc_biases_[i] : fc_biases_[128 + i];

            // Compute dot product with weights
            for (std::size_t j = 0; j < input_size; ++j) {
                std::size_t weight_idx = i * input_size + j;
                if (weight_idx < fc_weights_[layer_idx].size()) {
                    output[i] += input[j] * fc_weights_[layer_idx][weight_idx];
                }
            }
        }

        return output;
    }

    /**
     * @brief Apply softmax activation
     */
    void apply_softmax(std::vector<float>& logits) {
        float max_logit = *std::max_element(logits.begin(), logits.end());

        float sum_exp = 0.0f;
        for (auto& logit : logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }

        for (auto& prob : logits) {
            prob /= sum_exp;
        }
    }

  public:
    VisionTask get_specialization() const { return specialization_; }
    std::size_t get_expert_id() const { return expert_id_; }

    std::string get_specialization_name() const {
        switch (specialization_) {
            case VisionTask::OBJECT_DETECTION:
                return "Object Detection";
            case VisionTask::SCENE_CLASSIFICATION:
                return "Scene Classification";
            case VisionTask::FACIAL_RECOGNITION:
                return "Facial Recognition";
            default:
                return "Generic Vision";
        }
    }
};

// ================================================================================================
// MoE COMPUTER VISION SYSTEM
// ================================================================================================

/**
 * @brief Complete MoE-based computer vision system
 */
class MoEComputerVisionSystem {
  private:
    std::unique_ptr<MoEEngine> moe_engine_;
    std::vector<std::unique_ptr<ComputerVisionExpert>> experts_;
    MoEConfig config_;

    // Performance monitoring
    std::atomic<std::size_t> total_inferences_{0};
    std::atomic<std::size_t> correct_predictions_{0};
    std::map<VisionTask, std::size_t> task_counts_;
    std::chrono::steady_clock::time_point start_time_;

  public:
    explicit MoEComputerVisionSystem(const MoEConfig& config) : config_(config) {
        start_time_ = std::chrono::steady_clock::now();
        initialize_system();
    }

    /**
     * @brief Initialize the MoE system with specialized CV experts
     */
    void initialize_system() {
        std::vector<VisionTask> tasks = {VisionTask::OBJECT_DETECTION,
                                         VisionTask::SCENE_CLASSIFICATION,
                                         VisionTask::FACIAL_RECOGNITION};

        // Create multiple experts per task for redundancy and load balancing
        for (std::size_t expert_id = 0; expert_id < config_.num_experts; ++expert_id) {
            VisionTask task = tasks[expert_id % tasks.size()];
            experts_.push_back(std::make_unique<ComputerVisionExpert>(task, expert_id));
        }

        // Initialize MoE engine
        auto moe_result = MoEEngine::create(config_);
        if (moe_result.is_err()) {
            throw std::runtime_error("Failed to initialize MoE engine");
        }
        moe_engine_ = std::move(moe_result).unwrap();

        std::cout << "✅ Initialized MoE Computer Vision System with " << config_.num_experts
                  << " experts\n";

        // Print expert specializations
        std::map<VisionTask, int> task_counts;
        for (const auto& expert : experts_) {
            task_counts[expert->get_specialization()]++;
        }

        std::cout << "📊 Expert Specializations:\n";
        for (const auto& [task, count] : task_counts) {
            std::string task_name = task == VisionTask::OBJECT_DETECTION ? "Object Detection"
                                    : task == VisionTask::SCENE_CLASSIFICATION
                                        ? "Scene Classification"
                                    : task == VisionTask::FACIAL_RECOGNITION ? "Facial Recognition"
                                                                             : "Unknown";
            std::cout << "   • " << task_name << ": " << count << " experts\n";
        }
        std::cout << "\n";
    }

    /**
     * @brief Process image using MoE system
     */
    Result<VisionLabel, std::string> process_image(const ImageData& image,
                                                   VisionTask expected_task) {
        // Extract CNN features from image
        auto features = image.extract_cnn_features();

        // Prepare MoE input
        MoEInput moe_input;
        moe_input.features = features;
        moe_input.batch_size = 1;
        moe_input.enable_load_balancing = true;
        moe_input.request_id = total_inferences_.fetch_add(1);
        moe_input.priority = (expected_task == VisionTask::FACIAL_RECOGNITION) ? 1.5f : 1.0f;

        // Run MoE inference
        auto moe_result = moe_engine_->run_inference(moe_input);
        if (moe_result.is_err()) {
            return inference_lab::common::Result<VisionLabel, std::string>(
                inference_lab::common::Err<std::string>("MoE inference failed"));
        }

        auto response = std::move(moe_result).unwrap();

        // Execute selected experts and combine results
        std::vector<float> combined_probs(3, 0.0f);
        float total_weight = 0.0f;

        std::vector<std::string> selected_expert_names;

        for (std::size_t i = 0; i < response.selected_experts.size(); ++i) {
            std::size_t expert_id = response.selected_experts[i];
            float weight = response.expert_weights[i];

            if (expert_id < experts_.size()) {
                auto expert_output = experts_[expert_id]->forward(features);
                selected_expert_names.push_back(experts_[expert_id]->get_specialization_name());

                // Weight and combine expert predictions
                for (std::size_t j = 0; j < combined_probs.size() && j < expert_output.size();
                     ++j) {
                    combined_probs[j] += weight * expert_output[j];
                }
                total_weight += weight;
            }
        }

        // Normalize combined probabilities
        if (total_weight > 0.0f) {
            for (auto& prob : combined_probs) {
                prob /= total_weight;
            }
        }

        // Find predicted class
        auto max_it = std::max_element(combined_probs.begin(), combined_probs.end());
        VisionLabel predicted =
            static_cast<VisionLabel>(std::distance(combined_probs.begin(), max_it));

        // Print detailed inference results
        std::cout << "🔍 Computer Vision Inference:\n";
        std::cout << "   Selected Experts: ";
        for (const auto& name : selected_expert_names) {
            std::cout << name << " ";
        }
        std::cout << "\n   Expert Weights: ";
        for (auto weight : response.expert_weights) {
            std::cout << std::fixed << std::setprecision(3) << weight << " ";
        }
        std::cout << "\n   Class Probabilities: [";
        for (std::size_t i = 0; i < combined_probs.size(); ++i) {
            std::cout << std::fixed << std::setprecision(3) << combined_probs[i];
            if (i < combined_probs.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "   Routing Latency: " << response.routing_latency_ms << "ms\n";
        std::cout << "   Processing Latency: " << response.inference_latency_ms << "ms\n";
        std::cout << "   Active Parameters: " << response.active_parameters << "\n\n";

        // Update task statistics
        task_counts_[expected_task]++;

        return inference_lab::common::Result<VisionLabel, std::string>(
            inference_lab::common::Ok<VisionLabel>(predicted));
    }

    /**
     * @brief Print performance statistics
     */
    void print_statistics() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);

        std::size_t total = total_inferences_.load();
        std::size_t correct = correct_predictions_.load();

        double accuracy = total > 0 ? static_cast<double>(correct) / total * 100.0 : 0.0;
        double throughput = total > 0 ? static_cast<double>(total) / duration.count() * 1000.0
                                      : 0.0;

        std::cout << "📊 Computer Vision MoE Statistics:\n";
        std::cout << "   Total Images Processed: " << total << "\n";
        std::cout << "   Correct Classifications: " << correct << "\n";
        std::cout << "   Overall Accuracy: " << std::fixed << std::setprecision(1) << accuracy
                  << "%\n";
        std::cout << "   Processing Throughput: " << std::fixed << std::setprecision(1)
                  << throughput << " images/second\n";
        std::cout << "   Runtime: " << duration.count() << "ms\n";

        std::cout << "\n📈 Task Distribution:\n";
        for (const auto& [task, count] : task_counts_) {
            std::string task_name = task == VisionTask::OBJECT_DETECTION ? "Object Detection"
                                    : task == VisionTask::SCENE_CLASSIFICATION
                                        ? "Scene Classification"
                                    : task == VisionTask::FACIAL_RECOGNITION ? "Facial Recognition"
                                                                             : "Unknown";
            std::cout << "   • " << task_name << ": " << count << " images\n";
        }
        std::cout << "\n";
    }

    void record_prediction(bool correct) {
        if (correct) {
            correct_predictions_.fetch_add(1);
        }
    }
};

// ================================================================================================
// DEMONSTRATION DATA AND MAIN FUNCTION
// ================================================================================================

/**
 * @brief Create sample image data for demonstration
 */
std::vector<ImageSample> create_sample_images() {
    std::mt19937 gen(42);
    std::vector<ImageSample> samples;

    // Object detection samples
    samples.emplace_back(ImageData::generate_synthetic(gen, "object"),
                         VisionTask::OBJECT_DETECTION,
                         VisionLabel::PERSON,
                         "Person detection in street scene");

    samples.emplace_back(ImageData::generate_synthetic(gen, "object"),
                         VisionTask::OBJECT_DETECTION,
                         VisionLabel::CAR,
                         "Vehicle identification in parking lot");

    samples.emplace_back(ImageData::generate_synthetic(gen, "object"),
                         VisionTask::OBJECT_DETECTION,
                         VisionLabel::BUILDING,
                         "Building recognition in cityscape");

    // Scene classification samples
    samples.emplace_back(ImageData::generate_synthetic(gen, "scene"),
                         VisionTask::SCENE_CLASSIFICATION,
                         VisionLabel::INDOOR,
                         "Indoor office environment");

    samples.emplace_back(ImageData::generate_synthetic(gen, "scene"),
                         VisionTask::SCENE_CLASSIFICATION,
                         VisionLabel::OUTDOOR,
                         "Natural outdoor landscape");

    samples.emplace_back(ImageData::generate_synthetic(gen, "scene"),
                         VisionTask::SCENE_CLASSIFICATION,
                         VisionLabel::URBAN,
                         "Urban street scene");

    // Facial recognition samples
    samples.emplace_back(ImageData::generate_synthetic(gen, "face"),
                         VisionTask::FACIAL_RECOGNITION,
                         VisionLabel::FACE_DETECTED,
                         "Single face detection");

    samples.emplace_back(ImageData::generate_synthetic(gen, "face"),
                         VisionTask::FACIAL_RECOGNITION,
                         VisionLabel::NO_FACE,
                         "No faces in landscape image");

    samples.emplace_back(ImageData::generate_synthetic(gen, "face"),
                         VisionTask::FACIAL_RECOGNITION,
                         VisionLabel::MULTIPLE_FACES,
                         "Group photo with multiple faces");

    return samples;
}

/**
 * @brief Main demonstration function
 */
int main() {
    std::cout << "🖼️  MoE Computer Vision Demonstration\n";
    std::cout << "=====================================\n\n";

    // Configure MoE system for computer vision
    MoEConfig config;
    config.num_experts = 9;                // 3 experts per vision task
    config.expert_capacity = 2;            // Select top-2 experts
    config.load_balancing_weight = 0.15f;  // Slightly higher for load balancing
    config.enable_sparse_activation = true;
    config.max_concurrent_requests = 20;  // Higher for image processing

    try {
        // Initialize MoE computer vision system
        MoEComputerVisionSystem cv_system(config);

        // Create sample image data
        auto image_samples = create_sample_images();

        std::cout << "🧪 Running Computer Vision Classification Tests:\n";
        std::cout << "==============================================\n\n";

        // Process each image sample
        for (const auto& sample : image_samples) {
            std::cout << "🖼️  Processing: " << sample.description << "\n";
            std::cout << "🎯 Expected Task: "
                      << (sample.task == VisionTask::OBJECT_DETECTION       ? "Object Detection"
                          : sample.task == VisionTask::SCENE_CLASSIFICATION ? "Scene Classification"
                          : sample.task == VisionTask::FACIAL_RECOGNITION   ? "Facial Recognition"
                                                                            : "Unknown")
                      << "\n";
            std::cout << "🏷️  Expected Label: " << static_cast<int>(sample.label) << "\n";

            auto result = cv_system.process_image(sample.image, sample.task);
            if (result.is_ok()) {
                VisionLabel predicted = std::move(result).unwrap();
                bool correct = (predicted == sample.label);

                std::cout << "✨ Predicted Label: " << static_cast<int>(predicted)
                          << (correct ? " ✅ CORRECT" : " ❌ INCORRECT") << "\n";

                cv_system.record_prediction(correct);
            } else {
                std::cout << "❌ Processing failed: " << std::move(result).unwrap_err() << "\n";
            }

            std::cout << "----------------------------------------\n\n";
        }

        // Print final statistics
        cv_system.print_statistics();

        std::cout << "✅ MoE Computer Vision Demo Complete!\n";
        std::cout << "\n🎨 Key Features Demonstrated:\n";
        std::cout << "   • Task-specific CNN expert architectures\n";
        std::cout << "   • Visual feature extraction and routing\n";
        std::cout << "   • Multi-task learning with specialized experts\n";
        std::cout << "   • Load balancing for high-throughput processing\n";
        std::cout << "   • Real-time performance monitoring\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
