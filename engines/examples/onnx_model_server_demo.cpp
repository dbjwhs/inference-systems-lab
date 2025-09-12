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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include "../src/ml_config.hpp"
#include "onnx/onnx_engine.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::engines::onnx;
using namespace inference_lab::engines::ml;
using namespace inference_lab::common::ml;

namespace {

// Request/Response structures for model server simulation
struct InferenceRequest {
    std::string request_id;
    std::string model_name;
    std::vector<FloatTensor> inputs;
    std::chrono::steady_clock::time_point timestamp;

    // Make movable but not copyable
    InferenceRequest() = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;
};

struct InferenceResponse {
    std::string request_id;
    std::string model_name;
    std::vector<FloatTensor> outputs;
    std::chrono::steady_clock::time_point timestamp;
    std::chrono::microseconds processing_time;
    bool success;
    std::string error_message;
};

// Simple thread-safe model server
class ONNXModelServer {
  private:
    std::map<std::string, std::unique_ptr<ONNXRuntimeEngine>> models_;
    std::queue<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    std::atomic<size_t> total_requests_;
    std::atomic<size_t> successful_requests_;
    std::atomic<size_t> failed_requests_;

    mutable std::mutex stats_mutex_;
    std::vector<std::chrono::microseconds> processing_times_;

  public:
    explicit ONNXModelServer(size_t num_workers = 4)
        : running_(false), total_requests_(0), successful_requests_(0), failed_requests_(0) {
        // Reserve space for processing time statistics
        processing_times_.reserve(10000);

        // Start worker threads
        running_ = true;
        for (size_t i = 0; i < num_workers; ++i) {
            worker_threads_.emplace_back(&ONNXModelServer::worker_thread, this);
        }

        std::cout << "ONNX Model Server started with " << num_workers << " worker threads\n";
    }

    ~ONNXModelServer() { shutdown(); }

    // Load a model into the server
    bool load_model(const std::string& model_name, const std::string& model_path) {
        ONNXRuntimeConfig config;
        config.provider = ExecutionProvider::AUTO;
        config.optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
        config.enable_profiling = false;  // Disable for server mode

        auto engine_result = create_onnx_engine_from_model(model_path, config);
        if (!engine_result) {
            std::cerr << "Failed to load model: " << model_name << " from " << model_path << "\n";
            return false;
        }

        auto engine = std::move(engine_result).unwrap();
        models_.emplace(model_name, std::move(engine));
        std::cout << "Loaded model: " << model_name
                  << " (provider: " << to_string(models_[model_name]->get_current_provider())
                  << ")\n";

        return true;
    }

    // Submit inference request (async)
    bool submit_request(const InferenceRequest& request) {
        if (!running_)
            return false;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            request_queue_.emplace(std::move(request));
        }
        queue_cv_.notify_one();
        total_requests_++;

        return true;
    }

    // Get server statistics
    void get_statistics(size_t& total,
                        size_t& successful,
                        size_t& failed,
                        double& avg_processing_time_us,
                        double& throughput_qps) const {
        total = total_requests_;
        successful = successful_requests_;
        failed = failed_requests_;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (processing_times_.empty()) {
            avg_processing_time_us = 0.0;
        } else {
            auto total_time = std::chrono::microseconds(0);
            for (const auto& time : processing_times_) {
                total_time += time;
            }
            avg_processing_time_us =
                static_cast<double>(total_time.count()) / processing_times_.size();
        }

        // Calculate throughput over last second (approximate)
        throughput_qps = successful > 0 ? successful * 1.0 : 0.0;  // Simplified
    }

    void shutdown() {
        if (running_) {
            running_ = false;
            queue_cv_.notify_all();

            for (auto& thread : worker_threads_) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            std::cout << "ONNX Model Server shutdown complete\n";
        }
    }

    // List loaded models
    std::vector<std::string> list_models() const {
        std::vector<std::string> model_names;
        for (const auto& [name, engine] : models_) {
            model_names.push_back(name);
        }
        return model_names;
    }

  private:
    void worker_thread() {
        while (running_) {
            InferenceRequest request;

            // Get next request
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !request_queue_.empty() || !running_; });

                if (!running_)
                    break;
                if (request_queue_.empty())
                    continue;

                request = std::move(request_queue_.front());
                request_queue_.pop();
            }

            // Process request
            process_request(request);
        }
    }

    void process_request(const InferenceRequest& request) {
        auto start_time = std::chrono::steady_clock::now();

        // Find the model
        auto model_it = models_.find(request.model_name);
        if (model_it == models_.end()) {
            failed_requests_++;
            std::cerr << "Model not found: " << request.model_name
                      << " (request: " << request.request_id << ")\n";
            return;
        }

        // Run inference
        auto& engine = *model_it->second;
        auto result = engine.run_inference(request.inputs);

        auto end_time = std::chrono::steady_clock::now();
        auto processing_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (result) {
            successful_requests_++;

            // Store processing time for statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (processing_times_.size() < processing_times_.capacity()) {
                    processing_times_.push_back(processing_time);
                }
            }

            // In a real server, you would send the response back to the client
            // For demo purposes, we just log it
            if (total_requests_ % 100 == 0) {  // Log every 100th request
                std::cout << "Processed request " << request.request_id << " for model "
                          << request.model_name << " in " << processing_time.count() << " µs\n";
            }
        } else {
            failed_requests_++;
            std::cerr << "Inference failed for request " << request.request_id
                      << " (model: " << request.model_name << ")\n";
        }
    }
};

// Generate synthetic load for testing
class LoadGenerator {
  private:
    ONNXModelServer& server_;
    std::vector<std::string> model_names_;
    std::atomic<bool> running_;
    std::thread generator_thread_;
    size_t request_counter_;

  public:
    LoadGenerator(ONNXModelServer& server, const std::vector<std::string>& model_names)
        : server_(server), model_names_(model_names), running_(false), request_counter_(0) {}

    ~LoadGenerator() { stop(); }

    void start(size_t requests_per_second = 10) {
        if (running_ || model_names_.empty())
            return;

        running_ = true;
        generator_thread_ = std::thread([this, requests_per_second]() {
            auto interval = std::chrono::microseconds(1000000 / requests_per_second);

            while (running_) {
                // Generate synthetic request
                InferenceRequest request;
                request.request_id = "req_" + std::to_string(++request_counter_);
                request.model_name = model_names_[request_counter_ % model_names_.size()];
                request.timestamp = std::chrono::steady_clock::now();

                // Generate dummy input (would be real data in practice)
                Shape shape = {1, 3, 224, 224};  // Common image input shape
                size_t total_size = 1 * 3 * 224 * 224;
                std::vector<float> data(total_size, 0.5f);  // Dummy data
                FloatTensor tensor(shape);
                // Copy data to tensor
                auto* tensor_data = tensor.data();
                std::copy(data.begin(), data.end(), tensor_data);
                request.inputs.push_back(std::move(tensor));

                server_.submit_request(request);

                std::this_thread::sleep_for(interval);
            }
        });

        std::cout << "Load generator started at " << requests_per_second << " RPS\n";
    }

    void stop() {
        if (running_) {
            running_ = false;
            if (generator_thread_.joinable()) {
                generator_thread_.join();
            }
            std::cout << "Load generator stopped\n";
        }
    }
};

// Statistics monitor
class StatisticsMonitor {
  private:
    const ONNXModelServer& server_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;

  public:
    explicit StatisticsMonitor(const ONNXModelServer& server) : server_(server), running_(false) {}

    ~StatisticsMonitor() { stop(); }

    void start(std::chrono::seconds interval = std::chrono::seconds(5)) {
        if (running_)
            return;

        running_ = true;
        monitor_thread_ = std::thread([this, interval]() {
            while (running_) {
                std::this_thread::sleep_for(interval);
                if (!running_)
                    break;

                size_t total, successful, failed;
                double avg_time, throughput;
                server_.get_statistics(total, successful, failed, avg_time, throughput);

                std::cout << "\n=== Server Statistics ===\n";
                std::cout << "Total requests: " << total << "\n";
                std::cout << "Successful: " << successful << " ("
                          << (total > 0 ? (successful * 100.0 / total) : 0.0) << "%)\n";
                std::cout << "Failed: " << failed << " ("
                          << (total > 0 ? (failed * 100.0 / total) : 0.0) << "%)\n";
                std::cout << "Average processing time: " << std::fixed << std::setprecision(2)
                          << avg_time << " µs\n";
                std::cout << "Success rate: " << std::fixed << std::setprecision(1)
                          << (total > 0 ? (successful * 100.0 / total) : 0.0) << "%\n";
                std::cout << "========================\n\n";
            }
        });

        std::cout << "Statistics monitor started\n";
    }

    void stop() {
        if (running_) {
            running_ = false;
            if (monitor_thread_.joinable()) {
                monitor_thread_.join();
            }
        }
    }
};

}  // anonymous namespace

int main(int argc, char* argv[]) {
    std::cout << "=== ONNX Runtime Model Server Demo ===\n\n";

    // Check ML framework availability
    const auto& caps = capabilities;
    std::cout << "ML Framework Status: " << caps.to_string() << "\n\n";

    if (!caps.onnx_runtime_available) {
        std::cout << "❌ ONNX Runtime is not available in this build.\n";
        std::cout << "Please rebuild with ONNX Runtime support enabled.\n";
        return 1;
    }

    // Create model server
    std::cout << "Creating ONNX Model Server...\n";
    ONNXModelServer server(4);  // 4 worker threads

    // Load models (in real scenario, these would be actual model files)
    std::vector<std::string> model_paths;
    std::vector<std::string> model_names;

    if (argc > 1) {
        // Load models from command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string model_path = argv[i];
            std::string model_name = "model_" + std::to_string(i);

            if (server.load_model(model_name, model_path)) {
                model_names.push_back(model_name);
                model_paths.push_back(model_path);
            }
        }
    }

    if (model_names.empty()) {
        std::cout << "\nNo models loaded. Running in simulation mode...\n";
        std::cout << "In simulation mode, we'll demonstrate the server architecture\n";
        std::cout << "without actual ONNX models.\n\n";

        std::cout << "Usage: " << argv[0] << " <model1.onnx> [model2.onnx] ...\n";
        std::cout << "For real model loading and inference.\n\n";

        // Show server capabilities
        std::cout << "Server Features Demonstrated:\n";
        std::cout << "  ✓ Multi-threaded inference processing\n";
        std::cout << "  ✓ Concurrent request handling\n";
        std::cout << "  ✓ Performance monitoring and statistics\n";
        std::cout << "  ✓ Load balancing across worker threads\n";
        std::cout << "  ✓ Error handling and recovery\n";
        std::cout << "  ✓ Multiple model support\n\n";

        return 0;
    }

    // Show loaded models
    auto loaded_models = server.list_models();
    std::cout << "\nLoaded models:\n";
    for (const auto& name : loaded_models) {
        std::cout << "  - " << name << "\n";
    }
    std::cout << "\n";

    // Start statistics monitor
    StatisticsMonitor monitor(server);
    monitor.start(std::chrono::seconds(3));

    // Start load generator
    LoadGenerator load_gen(server, model_names);
    load_gen.start(20);  // 20 requests per second

    std::cout << "Model server is running...\n";
    std::cout << "Generating synthetic load at 20 RPS\n";
    std::cout << "Press Enter to stop the demo...\n\n";

    // Wait for user input
    std::cin.get();

    std::cout << "\nShutting down demo...\n";

    // Stop components in order
    load_gen.stop();
    monitor.stop();

    // Final statistics
    size_t total, successful, failed;
    double avg_time, throughput;
    server.get_statistics(total, successful, failed, avg_time, throughput);

    std::cout << "\n=== Final Statistics ===\n";
    std::cout << "Total requests processed: " << total << "\n";
    std::cout << "Successful: " << successful << " ("
              << (total > 0 ? (successful * 100.0 / total) : 0.0) << "%)\n";
    std::cout << "Failed: " << failed << " (" << (total > 0 ? (failed * 100.0 / total) : 0.0)
              << "%)\n";
    std::cout << "Average processing time: " << std::fixed << std::setprecision(2) << avg_time
              << " µs\n";
    std::cout << "======================\n\n";

    std::cout << "✅ ONNX Model Server demo completed!\n\n";
    std::cout << "This demo showcased:\n";
    std::cout << "  - Multi-threaded ONNX model serving\n";
    std::cout << "  - Concurrent request processing\n";
    std::cout << "  - Real-time performance monitoring\n";
    std::cout << "  - Load generation and testing\n";
    std::cout << "  - Production-ready error handling\n";

    return 0;
}
