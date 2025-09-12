// MIT License
// Copyright (c) 2025 dbjwhs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <chrono>
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
// TEXT FEATURE EXTRACTION
// ================================================================================================

/**
 * @brief Simple text feature extraction for demonstration
 *
 * In a real application, this would use proper NLP preprocessing:
 * - Tokenization, stemming, stop word removal
 * - TF-IDF, word embeddings, or transformer features
 * - N-gram analysis, syntactic features
 */
struct TextFeatures {
    std::vector<float> word_frequency;      // Top 100 words frequency
    std::vector<float> char_features;       // Character-level features
    std::vector<float> syntactic_features;  // Sentence length, punctuation, etc.

    static constexpr std::size_t FEATURE_DIM = 256;

    /**
     * @brief Extract features from text (simplified demonstration)
     */
    static TextFeatures extract(const std::string& text) {
        TextFeatures features;
        features.word_frequency.resize(100, 0.0f);
        features.char_features.resize(100, 0.0f);
        features.syntactic_features.resize(56, 0.0f);

        // Simplified feature extraction for demo
        // Word frequency simulation
        std::hash<std::string> hasher;
        auto hash = hasher(text);

        for (std::size_t i = 0; i < features.word_frequency.size(); ++i) {
            features.word_frequency[i] = static_cast<float>((hash + i) % 1000) / 1000.0f;
        }

        // Character-level features (letter frequencies, punctuation, etc.)
        for (char c : text) {
            if (c >= 'a' && c <= 'z') {
                features.char_features[c - 'a'] += 1.0f;
            } else if (c >= 'A' && c <= 'Z') {
                features.char_features[c - 'A'] += 1.0f;
            }
        }

        // Normalize character frequencies
        float total_chars = static_cast<float>(text.length());
        if (total_chars > 0) {
            for (auto& freq : features.char_features) {
                freq /= total_chars;
            }
        }

        // Syntactic features
        features.syntactic_features[0] = static_cast<float>(text.length());  // Text length
        features.syntactic_features[1] =
            static_cast<float>(std::count(text.begin(), text.end(), '.'));
        features.syntactic_features[2] =
            static_cast<float>(std::count(text.begin(), text.end(), '?'));
        features.syntactic_features[3] =
            static_cast<float>(std::count(text.begin(), text.end(), '!'));
        features.syntactic_features[4] =
            static_cast<float>(std::count(text.begin(), text.end(), ','));

        return features;
    }

    /**
     * @brief Convert to flat feature vector for MoE input
     */
    std::vector<float> to_vector() const {
        std::vector<float> result;
        result.reserve(FEATURE_DIM);

        result.insert(result.end(), word_frequency.begin(), word_frequency.end());
        result.insert(result.end(), char_features.begin(), char_features.end());
        result.insert(result.end(), syntactic_features.begin(), syntactic_features.end());

        return result;
    }
};

// ================================================================================================
// TEXT CLASSIFICATION DOMAINS
// ================================================================================================

/**
 * @brief Text classification domains for expert specialization
 */
enum class TextDomain : std::uint8_t {
    NEWS = 0,       // News articles, journalism
    REVIEWS = 1,    // Product reviews, opinions
    TECHNICAL = 2,  // Technical documentation, papers
    SOCIAL = 3,     // Social media, informal text
    ACADEMIC = 4,   // Academic papers, formal writing
    UNKNOWN = 255
};

/**
 * @brief Classification labels within each domain
 */
enum class ClassificationLabel : std::uint8_t {
    POSITIVE = 0,
    NEGATIVE = 1,
    NEUTRAL = 2,
    CATEGORY_A = 0,  // For non-sentiment tasks
    CATEGORY_B = 1,
    CATEGORY_C = 2
};

/**
 * @brief Sample text data for demonstration
 */
struct TextSample {
    std::string text;
    TextDomain domain;
    ClassificationLabel label;

    TextSample(std::string t, TextDomain d, ClassificationLabel l)
        : text(std::move(t)), domain(d), label(l) {}
};

// ================================================================================================
// NEURAL NETWORK EXPERT SIMULATION
// ================================================================================================

/**
 * @brief Simulated neural network expert for text classification
 *
 * In a real implementation, this would interface with:
 * - PyTorch via TorchScript/LibTorch
 * - ONNX Runtime models
 * - TensorFlow Lite
 * - Custom neural network implementations
 */
class TextClassificationExpert {
  private:
    TextDomain specialization_;
    std::size_t expert_id_;
    std::vector<std::vector<float>> weights_layer1_;  // Input -> Hidden
    std::vector<std::vector<float>> weights_layer2_;  // Hidden -> Output
    std::vector<float> bias_layer1_;
    std::vector<float> bias_layer2_;

  public:
    explicit TextClassificationExpert(TextDomain specialization, std::size_t expert_id)
        : specialization_(specialization), expert_id_(expert_id) {
        initialize_network();
    }

    /**
     * @brief Initialize neural network weights (random for demo)
     */
    void initialize_network() {
        constexpr std::size_t INPUT_DIM = TextFeatures::FEATURE_DIM;
        constexpr std::size_t HIDDEN_DIM = 128;
        constexpr std::size_t OUTPUT_DIM = 3;  // 3 classes

        // Initialize with domain-specific random seed for different specializations
        std::mt19937 gen(42 + static_cast<std::size_t>(specialization_) * 1000 + expert_id_);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        // Layer 1: Input -> Hidden
        weights_layer1_.resize(HIDDEN_DIM);
        for (auto& row : weights_layer1_) {
            row.resize(INPUT_DIM);
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        bias_layer1_.resize(HIDDEN_DIM);
        for (auto& bias : bias_layer1_) {
            bias = dist(gen);
        }

        // Layer 2: Hidden -> Output
        weights_layer2_.resize(OUTPUT_DIM);
        for (auto& row : weights_layer2_) {
            row.resize(HIDDEN_DIM);
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        bias_layer2_.resize(OUTPUT_DIM);
        for (auto& bias : bias_layer2_) {
            bias = dist(gen);
        }
    }

    /**
     * @brief Forward pass through the neural network
     */
    std::vector<float> forward(const std::vector<float>& input) {
        // Layer 1: Input -> Hidden (ReLU activation)
        std::vector<float> hidden(weights_layer1_.size());
        for (std::size_t i = 0; i < weights_layer1_.size(); ++i) {
            float sum = bias_layer1_[i];
            for (std::size_t j = 0; j < input.size() && j < weights_layer1_[i].size(); ++j) {
                sum += weights_layer1_[i][j] * input[j];
            }
            hidden[i] = std::max(0.0f, sum);  // ReLU activation
        }

        // Layer 2: Hidden -> Output (Softmax will be applied later)
        std::vector<float> output(weights_layer2_.size());
        for (std::size_t i = 0; i < weights_layer2_.size(); ++i) {
            float sum = bias_layer2_[i];
            for (std::size_t j = 0; j < hidden.size(); ++j) {
                sum += weights_layer2_[i][j] * hidden[j];
            }
            output[i] = sum;
        }

        // Apply softmax for probability distribution
        apply_softmax(output);

        return output;
    }

    /**
     * @brief Apply softmax activation to convert logits to probabilities
     */
    void apply_softmax(std::vector<float>& logits) {
        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());

        // Subtract max and exponentiate
        float sum_exp = 0.0f;
        for (auto& logit : logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }

        // Normalize
        for (auto& prob : logits) {
            prob /= sum_exp;
        }
    }

    TextDomain get_specialization() const { return specialization_; }
    std::size_t get_expert_id() const { return expert_id_; }
};

// ================================================================================================
// MoE TEXT CLASSIFICATION SYSTEM
// ================================================================================================

/**
 * @brief Complete MoE-based text classification system
 */
class MoETextClassifier {
  private:
    std::unique_ptr<MoEEngine> moe_engine_;
    std::vector<std::unique_ptr<TextClassificationExpert>> experts_;
    MoEConfig config_;

    // Performance monitoring
    std::atomic<std::size_t> total_inferences_{0};
    std::atomic<std::size_t> correct_predictions_{0};
    std::chrono::steady_clock::time_point start_time_;

  public:
    explicit MoETextClassifier(const MoEConfig& config) : config_(config) {
        start_time_ = std::chrono::steady_clock::now();
        initialize_system();
    }

    /**
     * @brief Initialize the MoE system and expert networks
     */
    void initialize_system() {
        // Create expert networks for different domains
        std::vector<TextDomain> domains = {TextDomain::NEWS,
                                           TextDomain::REVIEWS,
                                           TextDomain::TECHNICAL,
                                           TextDomain::SOCIAL,
                                           TextDomain::ACADEMIC};

        // Create multiple experts per domain for load balancing
        for (std::size_t expert_id = 0; expert_id < config_.num_experts; ++expert_id) {
            TextDomain domain = domains[expert_id % domains.size()];
            experts_.push_back(std::make_unique<TextClassificationExpert>(domain, expert_id));
        }

        // Initialize MoE engine
        auto moe_result = MoEEngine::create(config_);
        if (moe_result.is_err()) {
            throw std::runtime_error("Failed to initialize MoE engine");
        }
        moe_engine_ = std::move(moe_result).unwrap();

        std::cout << "✅ Initialized MoE Text Classifier with " << config_.num_experts
                  << " experts\n";
    }

    /**
     * @brief Classify text using MoE system
     */
    Result<ClassificationLabel, std::string> classify_text(const std::string& text) {
        // Extract features
        auto features = TextFeatures::extract(text);
        auto feature_vector = features.to_vector();

        // Prepare MoE input
        MoEInput moe_input;
        moe_input.features = feature_vector;
        moe_input.batch_size = 1;
        moe_input.enable_load_balancing = true;
        moe_input.request_id = total_inferences_.fetch_add(1);

        // Run MoE inference
        auto moe_result = moe_engine_->run_inference(moe_input);
        if (moe_result.is_err()) {
            return inference_lab::common::Result<ClassificationLabel, std::string>(
                inference_lab::common::Err<std::string>("MoE inference failed"));
        }

        auto response = std::move(moe_result).unwrap();

        // The MoE response contains aggregated outputs from selected experts
        // For this demo, we'll simulate the expert execution and combine results
        std::vector<float> combined_probs(3, 0.0f);  // 3 classes
        float total_weight = 0.0f;

        // Execute selected experts and combine their outputs
        for (std::size_t i = 0; i < response.selected_experts.size(); ++i) {
            std::size_t expert_id = response.selected_experts[i];
            float weight = response.expert_weights[i];

            if (expert_id < experts_.size()) {
                auto expert_output = experts_[expert_id]->forward(feature_vector);

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
        ClassificationLabel predicted =
            static_cast<ClassificationLabel>(std::distance(combined_probs.begin(), max_it));

        // Print inference details
        std::cout << "📊 Inference Details:\n";
        std::cout << "   Selected Experts: ";
        for (auto expert_id : response.selected_experts) {
            std::cout << expert_id << " ";
        }
        std::cout << "\n   Expert Weights: ";
        for (auto weight : response.expert_weights) {
            std::cout << std::fixed << std::setprecision(3) << weight << " ";
        }
        std::cout << "\n   Routing Latency: " << response.routing_latency_ms << "ms\n";
        std::cout << "   Inference Latency: " << response.inference_latency_ms << "ms\n";
        std::cout << "   Active Parameters: " << response.active_parameters << "\n\n";

        return inference_lab::common::Result<ClassificationLabel, std::string>(
            inference_lab::common::Ok<ClassificationLabel>(predicted));
    }

    /**
     * @brief Get performance statistics
     */
    void print_statistics() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);

        std::size_t total = total_inferences_.load();
        std::size_t correct = correct_predictions_.load();

        double accuracy = total > 0 ? static_cast<double>(correct) / total * 100.0 : 0.0;
        double throughput = total > 0 ? static_cast<double>(total) / duration.count() * 1000.0
                                      : 0.0;

        std::cout << "📈 Performance Statistics:\n";
        std::cout << "   Total Inferences: " << total << "\n";
        std::cout << "   Correct Predictions: " << correct << "\n";
        std::cout << "   Accuracy: " << std::fixed << std::setprecision(1) << accuracy << "%\n";
        std::cout << "   Throughput: " << std::fixed << std::setprecision(1) << throughput
                  << " inferences/second\n";
        std::cout << "   Runtime: " << duration.count() << "ms\n\n";
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
 * @brief Create sample data for demonstration
 */
std::vector<TextSample> create_sample_data() {
    return {
        // News samples
        TextSample("Breaking news: Global markets surge on positive economic indicators and trade "
                   "agreements.",
                   TextDomain::NEWS,
                   ClassificationLabel::POSITIVE),
        TextSample(
            "Political tensions rise as controversial legislation passes through parliament.",
            TextDomain::NEWS,
            ClassificationLabel::NEGATIVE),
        TextSample("Local weather update: Partly cloudy skies expected throughout the week.",
                   TextDomain::NEWS,
                   ClassificationLabel::NEUTRAL),

        // Review samples
        TextSample("This product exceeded my expectations! Great quality and fast delivery.",
                   TextDomain::REVIEWS,
                   ClassificationLabel::POSITIVE),
        TextSample("Terrible experience. Poor customer service and defective product.",
                   TextDomain::REVIEWS,
                   ClassificationLabel::NEGATIVE),
        TextSample("The product is okay. Nothing special but does what it's supposed to do.",
                   TextDomain::REVIEWS,
                   ClassificationLabel::NEUTRAL),

        // Technical samples
        TextSample("The algorithm achieves O(log n) complexity through efficient tree-based "
                   "indexing structures.",
                   TextDomain::TECHNICAL,
                   ClassificationLabel::POSITIVE),
        TextSample("Memory leaks detected in the garbage collection subsystem require immediate "
                   "attention.",
                   TextDomain::TECHNICAL,
                   ClassificationLabel::NEGATIVE),
        TextSample("System specifications include 16GB RAM and dual-core processor architecture.",
                   TextDomain::TECHNICAL,
                   ClassificationLabel::NEUTRAL),

        // Social media samples
        TextSample("Just had the best coffee ever! ☕️ Perfect start to the day! #coffee #morning",
                   TextDomain::SOCIAL,
                   ClassificationLabel::POSITIVE),
        TextSample("Stuck in traffic again... 😤 This commute is getting worse every day.",
                   TextDomain::SOCIAL,
                   ClassificationLabel::NEGATIVE),
        TextSample("Weather looks decent today. Might go for a walk later.",
                   TextDomain::SOCIAL,
                   ClassificationLabel::NEUTRAL),

        // Academic samples
        TextSample("The research methodology demonstrates significant improvements in experimental "
                   "validation.",
                   TextDomain::ACADEMIC,
                   ClassificationLabel::POSITIVE),
        TextSample("Critical limitations in the data collection process undermine the study's "
                   "reliability.",
                   TextDomain::ACADEMIC,
                   ClassificationLabel::NEGATIVE),
        TextSample("The literature review covers relevant prior work in the field comprehensively.",
                   TextDomain::ACADEMIC,
                   ClassificationLabel::NEUTRAL),
    };
}

/**
 * @brief Main demonstration function
 */
int main() {
    std::cout << "🚀 MoE Text Classification Demonstration\n";
    std::cout << "=========================================\n\n";

    // Configure MoE system
    MoEConfig config;
    config.num_experts = 8;
    config.expert_capacity = 2;
    config.load_balancing_weight = 0.1f;
    config.enable_sparse_activation = true;
    config.max_concurrent_requests = 10;

    try {
        // Initialize MoE classifier
        MoETextClassifier classifier(config);

        // Create sample data
        auto samples = create_sample_data();

        std::cout << "🧪 Running Text Classification Tests:\n";
        std::cout << "=====================================\n\n";

        // Classify each sample
        for (const auto& sample : samples) {
            std::cout << "📄 Input Text: \"" << sample.text.substr(0, 60) << "...\"\n";
            std::cout << "🎯 Expected Domain: " << static_cast<int>(sample.domain) << "\n";
            std::cout << "🏷️  Expected Label: " << static_cast<int>(sample.label) << "\n";

            auto result = classifier.classify_text(sample.text);
            if (result.is_ok()) {
                ClassificationLabel predicted = std::move(result).unwrap();
                bool correct = (predicted == sample.label);

                std::cout << "✨ Predicted Label: " << static_cast<int>(predicted)
                          << (correct ? " ✅ CORRECT" : " ❌ INCORRECT") << "\n";

                classifier.record_prediction(correct);
            } else {
                std::cout << "❌ Classification failed: " << std::move(result).unwrap_err() << "\n";
            }

            std::cout << "----------------------------------------\n\n";
        }

        // Print final statistics
        classifier.print_statistics();

        std::cout << "✅ MoE Text Classification Demo Complete!\n";
        std::cout << "\n🔬 Key Features Demonstrated:\n";
        std::cout << "   • Domain-specific expert networks\n";
        std::cout << "   • Automatic routing based on text characteristics\n";
        std::cout << "   • Load balancing across expert networks\n";
        std::cout << "   • Real-time performance monitoring\n";
        std::cout << "   • Sparse activation for computational efficiency\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
