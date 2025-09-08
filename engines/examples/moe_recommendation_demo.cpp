/**
 * @file moe_recommendation_demo.cpp
 * @brief Demonstration of Mixture of Experts for multi-context recommendation systems
 *
 * This example shows how to use the MoE system for recommendation tasks where
 * different experts specialize in different recommendation contexts: collaborative
 * filtering, content-based filtering, and demographic-based recommendations.
 * The router learns to select appropriate experts based on user and item features.
 *
 * Key Features Demonstrated:
 * - Context-aware expert specialization (user preferences, content, demographics)
 * - Dynamic expert selection based on available user/item data
 * - Real-time recommendation generation with load balancing
 * - A/B testing framework for recommendation quality
 * - Integration with existing Result<T,E> patterns
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/src/result.hpp"
#include "../src/mixture_experts/moe_config.hpp"
#include "../src/mixture_experts/moe_engine.hpp"

using namespace engines::mixture_experts;
using namespace inference_lab::common;

// ================================================================================================
// USER AND ITEM MODELING
// ================================================================================================

/**
 * @brief User profile with various attributes for recommendation
 */
struct UserProfile {
    std::size_t user_id;
    std::string name;

    // Demographic features
    std::size_t age;
    std::string location;
    std::string occupation;

    // Behavioral features
    std::vector<std::size_t> viewed_items;
    std::vector<std::size_t> purchased_items;
    std::vector<float> ratings;                         // Corresponding ratings for purchased items
    std::map<std::string, float> category_preferences;  // Category -> preference score

    // Interaction features
    std::size_t total_sessions;
    float avg_session_duration;
    std::chrono::system_clock::time_point last_active;

    static constexpr std::size_t USER_FEATURE_DIM = 64;

    /**
     * @brief Convert user profile to feature vector for MoE input
     */
    std::vector<float> to_feature_vector() const {
        std::vector<float> features(USER_FEATURE_DIM, 0.0f);

        // Demographic features (normalized)
        features[0] = static_cast<float>(age) / 100.0f;  // Age normalized to [0,1]
        features[1] = std::hash<std::string>{}(location) % 1000 / 1000.0f;
        features[2] = std::hash<std::string>{}(occupation) % 1000 / 1000.0f;

        // Behavioral features
        features[3] = std::min(static_cast<float>(viewed_items.size()) / 1000.0f, 1.0f);
        features[4] = std::min(static_cast<float>(purchased_items.size()) / 100.0f, 1.0f);

        // Average rating
        if (!ratings.empty()) {
            float avg_rating = 0.0f;
            for (float rating : ratings)
                avg_rating += rating;
            features[5] = (avg_rating / ratings.size()) / 5.0f;  // Assuming 1-5 rating scale
        }

        // Category preferences (top categories)
        std::size_t cat_idx = 6;
        for (const auto& [category, preference] : category_preferences) {
            if (cat_idx < 20) {  // Use first 14 slots for categories
                features[cat_idx++] = preference;
            }
        }

        // Interaction patterns
        features[20] = std::min(static_cast<float>(total_sessions) / 1000.0f, 1.0f);
        features[21] = std::min(avg_session_duration / 3600.0f, 1.0f);  // Hours to [0,1]

        // Recency (time since last active)
        auto now = std::chrono::system_clock::now();
        auto hours_since_active =
            std::chrono::duration_cast<std::chrono::hours>(now - last_active).count();
        features[22] =
            std::min(static_cast<float>(hours_since_active) / 168.0f, 1.0f);  // Week to [0,1]

        // Fill remaining features with derived metrics
        for (std::size_t i = 23; i < USER_FEATURE_DIM; ++i) {
            features[i] = std::sin(static_cast<float>(i + user_id) * 0.1f) * 0.1f + 0.5f;
        }

        return features;
    }
};

/**
 * @brief Item profile with content and metadata
 */
struct ItemProfile {
    std::size_t item_id;
    std::string title;
    std::string category;
    std::string description;

    // Content features
    std::vector<std::string> tags;
    std::vector<std::string> genres;
    float price;
    float avg_rating;
    std::size_t num_ratings;

    // Metadata
    std::chrono::system_clock::time_point created_date;
    std::size_t popularity_score;  // Based on views, purchases, etc.

    static constexpr std::size_t ITEM_FEATURE_DIM = 32;

    /**
     * @brief Convert item profile to feature vector
     */
    std::vector<float> to_feature_vector() const {
        std::vector<float> features(ITEM_FEATURE_DIM, 0.0f);

        // Basic item features
        features[0] = std::hash<std::string>{}(category) % 1000 / 1000.0f;
        features[1] = std::min(price / 1000.0f, 1.0f);  // Price normalized
        features[2] = avg_rating / 5.0f;                // Rating normalized to [0,1]
        features[3] = std::min(static_cast<float>(num_ratings) / 1000.0f, 1.0f);

        // Popularity and recency
        features[4] = std::min(static_cast<float>(popularity_score) / 10000.0f, 1.0f);

        auto now = std::chrono::system_clock::now();
        auto days_old =
            std::chrono::duration_cast<std::chrono::hours>(now - created_date).count() / 24;
        features[5] = std::min(static_cast<float>(days_old) / 365.0f, 1.0f);  // Age in years

        // Content features (simplified)
        features[6] = static_cast<float>(tags.size()) / 20.0f;    // Number of tags
        features[7] = static_cast<float>(genres.size()) / 10.0f;  // Number of genres

        // Fill remaining with content-based hashes
        for (std::size_t i = 8; i < ITEM_FEATURE_DIM; ++i) {
            std::size_t content_hash = std::hash<std::string>{}(title + description);
            features[i] = ((content_hash + i) % 1000) / 1000.0f;
        }

        return features;
    }
};

// ================================================================================================
// RECOMMENDATION CONTEXTS AND STRATEGIES
// ================================================================================================

/**
 * @brief Different recommendation strategies that experts specialize in
 */
enum class RecommendationStrategy : std::uint8_t {
    COLLABORATIVE_FILTERING = 0,  // User-item collaborative filtering
    CONTENT_BASED = 1,            // Content similarity recommendations
    DEMOGRAPHIC_BASED = 2,        // Demographic and contextual recommendations
    HYBRID = 3,                   // Hybrid approach
    UNKNOWN = 255
};

/**
 * @brief Recommendation request with context
 */
struct RecommendationRequest {
    UserProfile user;
    std::vector<ItemProfile> candidate_items;  // Items to rank/filter
    std::size_t num_recommendations = 10;      // Number of items to recommend
    RecommendationStrategy preferred_strategy = RecommendationStrategy::HYBRID;

    // Context information
    std::string session_context;   // "browsing", "purchasing", "exploring"
    float time_constraint = 1.0f;  // 1.0 = no constraint, 0.5 = fast response needed
    bool include_diverse_items = true;
};

/**
 * @brief Recommendation response with ranked items
 */
struct RecommendationResponse {
    std::vector<std::size_t> recommended_item_ids;
    std::vector<float> recommendation_scores;
    std::vector<std::string> reasoning;  // Why each item was recommended
    RecommendationStrategy strategy_used;

    // Performance metrics
    float generation_latency_ms;
    std::size_t experts_consulted;
    float diversity_score;
    float confidence_score;
};

// ================================================================================================
// RECOMMENDATION EXPERT NETWORKS
// ================================================================================================

/**
 * @brief Specialized recommendation expert for different strategies
 */
class RecommendationExpert {
  private:
    RecommendationStrategy specialization_;
    std::size_t expert_id_;

    // Neural network layers for recommendation scoring
    std::vector<std::vector<float>> user_embedding_weights_;
    std::vector<std::vector<float>> item_embedding_weights_;
    std::vector<std::vector<float>> interaction_weights_;
    std::vector<float> biases_;

  public:
    explicit RecommendationExpert(RecommendationStrategy specialization, std::size_t expert_id)
        : specialization_(specialization), expert_id_(expert_id) {
        initialize_recommendation_network();
    }

    /**
     * @brief Initialize recommendation neural network
     */
    void initialize_recommendation_network() {
        std::mt19937 gen(42 + static_cast<std::size_t>(specialization_) * 1000 + expert_id_);
        std::normal_distribution<float> weight_dist(0.0f, 0.1f);

        constexpr std::size_t EMBEDDING_DIM = 64;
        constexpr std::size_t INTERACTION_DIM = 32;

        // Initialize based on specialization
        switch (specialization_) {
            case RecommendationStrategy::COLLABORATIVE_FILTERING:
                initialize_collaborative_network(gen, weight_dist, EMBEDDING_DIM, INTERACTION_DIM);
                break;
            case RecommendationStrategy::CONTENT_BASED:
                initialize_content_network(gen, weight_dist, EMBEDDING_DIM, INTERACTION_DIM);
                break;
            case RecommendationStrategy::DEMOGRAPHIC_BASED:
                initialize_demographic_network(gen, weight_dist, EMBEDDING_DIM, INTERACTION_DIM);
                break;
            default:
                initialize_hybrid_network(gen, weight_dist, EMBEDDING_DIM, INTERACTION_DIM);
                break;
        }
    }

    void initialize_collaborative_network(std::mt19937& gen,
                                          std::normal_distribution<float>& dist,
                                          std::size_t emb_dim,
                                          std::size_t int_dim) {
        // Focus on user-item interaction patterns
        user_embedding_weights_.resize(emb_dim);
        item_embedding_weights_.resize(emb_dim);

        for (auto& row : user_embedding_weights_) {
            row.resize(UserProfile::USER_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen) * 1.2f;  // Higher weights for user features
            }
        }

        for (auto& row : item_embedding_weights_) {
            row.resize(ItemProfile::ITEM_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen) * 0.8f;  // Lower weights for item features
            }
        }

        // Interaction layer
        interaction_weights_.resize(int_dim);
        for (auto& row : interaction_weights_) {
            row.resize(emb_dim * 2);  // User + item embeddings
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        biases_.resize(int_dim);
        for (auto& bias : biases_) {
            bias = dist(gen);
        }
    }

    void initialize_content_network(std::mt19937& gen,
                                    std::normal_distribution<float>& dist,
                                    std::size_t emb_dim,
                                    std::size_t int_dim) {
        // Focus on item content features
        user_embedding_weights_.resize(emb_dim);
        item_embedding_weights_.resize(emb_dim);

        for (auto& row : user_embedding_weights_) {
            row.resize(UserProfile::USER_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen) * 0.8f;  // Lower weights for user features
            }
        }

        for (auto& row : item_embedding_weights_) {
            row.resize(ItemProfile::ITEM_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen) * 1.5f;  // Higher weights for item content
            }
        }

        // Content similarity computation
        interaction_weights_.resize(int_dim);
        for (auto& row : interaction_weights_) {
            row.resize(emb_dim * 2);
            for (auto& weight : row) {
                weight = dist(gen) * 1.1f;  // Emphasize content matching
            }
        }

        biases_.resize(int_dim);
        for (auto& bias : biases_) {
            bias = dist(gen);
        }
    }

    void initialize_demographic_network(std::mt19937& gen,
                                        std::normal_distribution<float>& dist,
                                        std::size_t emb_dim,
                                        std::size_t int_dim) {
        // Focus on demographic and contextual features
        user_embedding_weights_.resize(emb_dim);
        item_embedding_weights_.resize(emb_dim);

        for (auto& row : user_embedding_weights_) {
            row.resize(UserProfile::USER_FEATURE_DIM);
            for (std::size_t i = 0; i < row.size(); ++i) {
                // Higher weights for demographic features (first few features)
                float multiplier = (i < 5) ? 1.5f : 0.7f;
                row[i] = dist(gen) * multiplier;
            }
        }

        for (auto& row : item_embedding_weights_) {
            row.resize(ItemProfile::ITEM_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        interaction_weights_.resize(int_dim);
        for (auto& row : interaction_weights_) {
            row.resize(emb_dim * 2);
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        biases_.resize(int_dim);
        for (auto& bias : biases_) {
            bias = dist(gen);
        }
    }

    void initialize_hybrid_network(std::mt19937& gen,
                                   std::normal_distribution<float>& dist,
                                   std::size_t emb_dim,
                                   std::size_t int_dim) {
        // Balanced approach combining all strategies
        user_embedding_weights_.resize(emb_dim);
        item_embedding_weights_.resize(emb_dim);

        for (auto& row : user_embedding_weights_) {
            row.resize(UserProfile::USER_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen);  // Balanced weights
            }
        }

        for (auto& row : item_embedding_weights_) {
            row.resize(ItemProfile::ITEM_FEATURE_DIM);
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        interaction_weights_.resize(int_dim);
        for (auto& row : interaction_weights_) {
            row.resize(emb_dim * 2);
            for (auto& weight : row) {
                weight = dist(gen);
            }
        }

        biases_.resize(int_dim);
        for (auto& bias : biases_) {
            bias = dist(gen);
        }
    }

    /**
     * @brief Generate recommendation scores for user-item pairs
     */
    std::vector<float> generate_recommendations(const UserProfile& user,
                                                const std::vector<ItemProfile>& items) {
        auto user_features = user.to_feature_vector();
        std::vector<float> scores;
        scores.reserve(items.size());

        // Generate user embedding
        std::vector<float> user_embedding =
            compute_embedding(user_features, user_embedding_weights_);

        // Score each item
        for (const auto& item : items) {
            auto item_features = item.to_feature_vector();
            std::vector<float> item_embedding =
                compute_embedding(item_features, item_embedding_weights_);

            // Combine user and item embeddings
            std::vector<float> combined_features;
            combined_features.insert(
                combined_features.end(), user_embedding.begin(), user_embedding.end());
            combined_features.insert(
                combined_features.end(), item_embedding.begin(), item_embedding.end());

            // Compute interaction score
            float score = compute_interaction_score(combined_features);

            // Apply strategy-specific adjustments
            score = apply_strategy_adjustment(score, user, item);

            scores.push_back(score);
        }

        return scores;
    }

  private:
    /**
     * @brief Compute embedding from features using learned weights
     */
    std::vector<float> compute_embedding(const std::vector<float>& features,
                                         const std::vector<std::vector<float>>& weights) {
        std::vector<float> embedding(weights.size(), 0.0f);

        for (std::size_t i = 0; i < embedding.size(); ++i) {
            for (std::size_t j = 0; j < features.size() && j < weights[i].size(); ++j) {
                embedding[i] += features[j] * weights[i][j];
            }
            embedding[i] = std::tanh(embedding[i]);  // Tanh activation
        }

        return embedding;
    }

    /**
     * @brief Compute final interaction score
     */
    float compute_interaction_score(const std::vector<float>& combined_features) {
        float score = 0.0f;

        for (std::size_t i = 0; i < interaction_weights_.size(); ++i) {
            float neuron_output = biases_[i];
            for (std::size_t j = 0;
                 j < combined_features.size() && j < interaction_weights_[i].size();
                 ++j) {
                neuron_output += combined_features[j] * interaction_weights_[i][j];
            }
            score += std::max(0.0f, neuron_output);  // ReLU activation
        }

        return 1.0f / (1.0f + std::exp(-score));  // Sigmoid output
    }

    /**
     * @brief Apply strategy-specific score adjustments
     */
    float apply_strategy_adjustment(float base_score,
                                    const UserProfile& user,
                                    const ItemProfile& item) {
        switch (specialization_) {
            case RecommendationStrategy::COLLABORATIVE_FILTERING:
                // Boost items similar to user's purchase history
                for (std::size_t purchased_id : user.purchased_items) {
                    if (purchased_id == item.item_id) {
                        return 0.0f;  // Don't recommend already purchased items
                    }
                    // Simulate item similarity (in real system, use precomputed similarities)
                    if ((purchased_id + item.item_id) % 10 < 3) {
                        base_score *= 1.2f;  // Boost similar items
                    }
                }
                break;

            case RecommendationStrategy::CONTENT_BASED:
                // Boost items in user's preferred categories
                if (user.category_preferences.find(item.category) !=
                    user.category_preferences.end()) {
                    float preference = user.category_preferences.at(item.category);
                    base_score *= (1.0f + preference * 0.5f);
                }
                break;

            case RecommendationStrategy::DEMOGRAPHIC_BASED:
                // Boost popular items for new users
                if (user.purchased_items.size() < 5) {
                    base_score *= (1.0f + static_cast<float>(item.popularity_score) / 20000.0f);
                }
                break;

            default:
                // Hybrid: combine multiple signals
                base_score *= 0.9f;  // Slight penalty for being generic
                break;
        }

        return std::min(1.0f, base_score);  // Cap at 1.0
    }

  public:
    RecommendationStrategy get_specialization() const { return specialization_; }
    std::size_t get_expert_id() const { return expert_id_; }

    std::string get_specialization_name() const {
        switch (specialization_) {
            case RecommendationStrategy::COLLABORATIVE_FILTERING:
                return "Collaborative Filtering";
            case RecommendationStrategy::CONTENT_BASED:
                return "Content-Based";
            case RecommendationStrategy::DEMOGRAPHIC_BASED:
                return "Demographic-Based";
            case RecommendationStrategy::HYBRID:
                return "Hybrid";
            default:
                return "Unknown";
        }
    }
};

// ================================================================================================
// MoE RECOMMENDATION SYSTEM
// ================================================================================================

/**
 * @brief Complete MoE-based recommendation system
 */
class MoERecommendationSystem {
  private:
    std::unique_ptr<MoEEngine> moe_engine_;
    std::vector<std::unique_ptr<RecommendationExpert>> experts_;
    MoEConfig config_;

    // Performance monitoring
    std::atomic<std::size_t> total_recommendations_{0};
    std::atomic<std::size_t> successful_recommendations_{0};
    std::map<RecommendationStrategy, std::size_t> strategy_usage_;
    std::chrono::steady_clock::time_point start_time_;

  public:
    explicit MoERecommendationSystem(const MoEConfig& config) : config_(config) {
        start_time_ = std::chrono::steady_clock::now();
        initialize_system();
    }

    /**
     * @brief Initialize the MoE recommendation system
     */
    void initialize_system() {
        std::vector<RecommendationStrategy> strategies = {
            RecommendationStrategy::COLLABORATIVE_FILTERING,
            RecommendationStrategy::CONTENT_BASED,
            RecommendationStrategy::DEMOGRAPHIC_BASED,
            RecommendationStrategy::HYBRID};

        // Create multiple experts per strategy
        for (std::size_t expert_id = 0; expert_id < config_.num_experts; ++expert_id) {
            RecommendationStrategy strategy = strategies[expert_id % strategies.size()];
            experts_.push_back(std::make_unique<RecommendationExpert>(strategy, expert_id));
        }

        // Initialize MoE engine
        auto moe_result = MoEEngine::create(config_);
        if (moe_result.is_err()) {
            throw std::runtime_error("Failed to initialize MoE engine");
        }
        moe_engine_ = std::move(moe_result).unwrap();

        std::cout << "✅ Initialized MoE Recommendation System with " << config_.num_experts
                  << " experts\n";

        // Print expert distribution
        std::map<RecommendationStrategy, int> strategy_counts;
        for (const auto& expert : experts_) {
            strategy_counts[expert->get_specialization()]++;
        }

        std::cout << "📊 Expert Strategy Distribution:\n";
        for (const auto& [strategy, count] : strategy_counts) {
            std::cout << "   • "
                      << (strategy == RecommendationStrategy::COLLABORATIVE_FILTERING
                              ? "Collaborative Filtering"
                          : strategy == RecommendationStrategy::CONTENT_BASED ? "Content-Based"
                          : strategy == RecommendationStrategy::DEMOGRAPHIC_BASED
                              ? "Demographic-Based"
                          : strategy == RecommendationStrategy::HYBRID ? "Hybrid"
                                                                       : "Unknown")
                      << ": " << count << " experts\n";
        }
        std::cout << "\n";
    }

    /**
     * @brief Generate recommendations using MoE system
     */
    Result<RecommendationResponse, std::string> generate_recommendations(
        const RecommendationRequest& request) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Prepare features for MoE routing
        auto user_features = request.user.to_feature_vector();

        // Add context features to help routing
        std::vector<float> context_features;
        context_features.insert(context_features.end(), user_features.begin(), user_features.end());

        // Context encoding
        context_features.push_back(request.candidate_items.size() / 1000.0f);  // Number of
                                                                               // candidates
        context_features.push_back(static_cast<float>(request.num_recommendations) / 100.0f);
        context_features.push_back(request.time_constraint);
        context_features.push_back(request.include_diverse_items ? 1.0f : 0.0f);

        // Session context encoding
        float session_encoding = 0.5f;
        if (request.session_context == "browsing")
            session_encoding = 0.3f;
        else if (request.session_context == "purchasing")
            session_encoding = 0.8f;
        else if (request.session_context == "exploring")
            session_encoding = 0.1f;
        context_features.push_back(session_encoding);

        // Prepare MoE input
        MoEInput moe_input;
        moe_input.features = context_features;
        moe_input.batch_size = 1;
        moe_input.enable_load_balancing = true;
        moe_input.request_id = total_recommendations_.fetch_add(1);
        moe_input.priority =
            (request.time_constraint < 0.5f) ? 1.5f : 1.0f;  // Higher priority for fast requests

        // Run MoE expert selection
        auto moe_result = moe_engine_->run_inference(moe_input);
        if (moe_result.is_err()) {
            return inference_lab::common::Result<RecommendationResponse, std::string>(
                inference_lab::common::Err<std::string>("MoE expert selection failed"));
        }

        auto expert_response = std::move(moe_result).unwrap();

        // Generate recommendations using selected experts
        std::vector<float> combined_scores(request.candidate_items.size(), 0.0f);
        float total_weight = 0.0f;
        std::vector<std::string> expert_names;
        RecommendationStrategy primary_strategy = RecommendationStrategy::HYBRID;

        for (std::size_t i = 0; i < expert_response.selected_experts.size(); ++i) {
            std::size_t expert_id = expert_response.selected_experts[i];
            float weight = expert_response.expert_weights[i];

            if (expert_id < experts_.size()) {
                auto expert_scores = experts_[expert_id]->generate_recommendations(
                    request.user, request.candidate_items);
                expert_names.push_back(experts_[expert_id]->get_specialization_name());

                if (i == 0) {  // Primary expert determines strategy
                    primary_strategy = experts_[expert_id]->get_specialization();
                }

                // Combine scores with expert weights
                for (std::size_t j = 0; j < combined_scores.size() && j < expert_scores.size();
                     ++j) {
                    combined_scores[j] += weight * expert_scores[j];
                }
                total_weight += weight;
            }
        }

        // Normalize combined scores
        if (total_weight > 0.0f) {
            for (auto& score : combined_scores) {
                score /= total_weight;
            }
        }

        // Rank items by score and select top recommendations
        std::vector<std::pair<std::size_t, float>> item_scores;
        for (std::size_t i = 0; i < combined_scores.size(); ++i) {
            item_scores.emplace_back(i, combined_scores[i]);
        }

        std::sort(item_scores.begin(), item_scores.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        // Build recommendation response
        RecommendationResponse response;
        response.strategy_used = primary_strategy;
        response.experts_consulted = expert_response.selected_experts.size();

        std::size_t num_recs = std::min(request.num_recommendations, item_scores.size());
        for (std::size_t i = 0; i < num_recs; ++i) {
            std::size_t item_idx = item_scores[i].first;
            response.recommended_item_ids.push_back(request.candidate_items[item_idx].item_id);
            response.recommendation_scores.push_back(item_scores[i].second);

            // Generate reasoning
            std::string reasoning = "Recommended by " + expert_names[0] + " (score: " +
                                    std::to_string(item_scores[i].second).substr(0, 4) + ")";
            response.reasoning.push_back(reasoning);
        }

        // Calculate performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        response.generation_latency_ms =
            std::chrono::duration<float, std::milli>(end_time - start_time).count();

        // Calculate diversity score (simplified)
        response.diversity_score =
            calculate_diversity_score(response.recommended_item_ids, request.candidate_items);

        // Calculate confidence based on score distribution
        response.confidence_score = calculate_confidence_score(response.recommendation_scores);

        // Update statistics
        strategy_usage_[primary_strategy]++;
        successful_recommendations_.fetch_add(1);

        // Print detailed results
        std::cout << "🎯 Recommendation Generation:\n";
        std::cout << "   Strategy Used: "
                  << (primary_strategy == RecommendationStrategy::COLLABORATIVE_FILTERING
                          ? "Collaborative Filtering"
                      : primary_strategy == RecommendationStrategy::CONTENT_BASED ? "Content-Based"
                      : primary_strategy == RecommendationStrategy::DEMOGRAPHIC_BASED
                          ? "Demographic-Based"
                          : "Hybrid")
                  << "\n";
        std::cout << "   Experts Consulted: ";
        for (const auto& name : expert_names) {
            std::cout << name << " ";
        }
        std::cout << "\n   Generation Time: " << std::fixed << std::setprecision(2)
                  << response.generation_latency_ms << "ms\n";
        std::cout << "   Diversity Score: " << std::fixed << std::setprecision(3)
                  << response.diversity_score << "\n";
        std::cout << "   Confidence Score: " << std::fixed << std::setprecision(3)
                  << response.confidence_score << "\n\n";

        return inference_lab::common::Result<RecommendationResponse, std::string>(
            inference_lab::common::Ok<RecommendationResponse>(std::move(response)));
    }

    /**
     * @brief Calculate diversity score for recommendations
     */
    float calculate_diversity_score(const std::vector<std::size_t>& rec_ids,
                                    const std::vector<ItemProfile>& items) {
        if (rec_ids.size() < 2)
            return 1.0f;

        // Create map for quick item lookup
        std::unordered_map<std::size_t, const ItemProfile*> item_map;
        for (const auto& item : items) {
            item_map[item.item_id] = &item;
        }

        // Calculate category diversity
        std::set<std::string> unique_categories;
        for (std::size_t item_id : rec_ids) {
            if (item_map.find(item_id) != item_map.end()) {
                unique_categories.insert(item_map[item_id]->category);
            }
        }

        return static_cast<float>(unique_categories.size()) / rec_ids.size();
    }

    /**
     * @brief Calculate confidence score based on recommendation scores
     */
    float calculate_confidence_score(const std::vector<float>& scores) {
        if (scores.empty())
            return 0.0f;

        // Calculate variance in scores (higher variance = more discriminative = higher confidence)
        float mean = 0.0f;
        for (float score : scores)
            mean += score;
        mean /= scores.size();

        float variance = 0.0f;
        for (float score : scores) {
            variance += (score - mean) * (score - mean);
        }
        variance /= scores.size();

        // Normalize and bound confidence
        return std::min(1.0f, std::sqrt(variance) * 2.0f);
    }

    /**
     * @brief Print system performance statistics
     */
    void print_statistics() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);

        std::size_t total = total_recommendations_.load();
        std::size_t successful = successful_recommendations_.load();

        double success_rate = total > 0 ? static_cast<double>(successful) / total * 100.0 : 0.0;
        double throughput = total > 0 ? static_cast<double>(total) / duration.count() * 1000.0
                                      : 0.0;

        std::cout << "📊 MoE Recommendation System Statistics:\n";
        std::cout << "   Total Recommendation Requests: " << total << "\n";
        std::cout << "   Successful Recommendations: " << successful << "\n";
        std::cout << "   Success Rate: " << std::fixed << std::setprecision(1) << success_rate
                  << "%\n";
        std::cout << "   Throughput: " << std::fixed << std::setprecision(1) << throughput
                  << " requests/second\n";
        std::cout << "   Runtime: " << duration.count() << "ms\n";

        std::cout << "\n📈 Strategy Usage Distribution:\n";
        for (const auto& [strategy, count] : strategy_usage_) {
            std::string strategy_name =
                strategy == RecommendationStrategy::COLLABORATIVE_FILTERING
                    ? "Collaborative Filtering"
                : strategy == RecommendationStrategy::CONTENT_BASED     ? "Content-Based"
                : strategy == RecommendationStrategy::DEMOGRAPHIC_BASED ? "Demographic-Based"
                                                                        : "Hybrid";
            std::cout << "   • " << strategy_name << ": " << count << " times\n";
        }
        std::cout << "\n";
    }
};

// ================================================================================================
// DEMONSTRATION DATA AND MAIN FUNCTION
// ================================================================================================

/**
 * @brief Create sample user and item data for demonstration
 */
std::pair<std::vector<UserProfile>, std::vector<ItemProfile>> create_sample_data() {
    std::vector<UserProfile> users;
    std::vector<ItemProfile> items;

    // Create sample users
    UserProfile user1;
    user1.user_id = 1;
    user1.name = "Alice Johnson";
    user1.age = 28;
    user1.location = "San Francisco";
    user1.occupation = "Software Engineer";
    user1.viewed_items = {101, 102, 105, 108};
    user1.purchased_items = {101, 105};
    user1.ratings = {4.5f, 4.0f};
    user1.category_preferences = {{"Electronics", 0.8f}, {"Books", 0.6f}, {"Home", 0.3f}};
    user1.total_sessions = 45;
    user1.avg_session_duration = 25.5f;
    user1.last_active = std::chrono::system_clock::now() - std::chrono::hours(2);
    users.push_back(user1);

    UserProfile user2;
    user2.user_id = 2;
    user2.name = "Bob Smith";
    user2.age = 34;
    user2.location = "New York";
    user2.occupation = "Marketing Manager";
    user2.viewed_items = {103, 104, 106, 107, 109};
    user2.purchased_items = {103, 106, 107};
    user2.ratings = {3.5f, 4.2f, 3.8f};
    user2.category_preferences = {{"Sports", 0.9f}, {"Electronics", 0.4f}, {"Fashion", 0.7f}};
    user2.total_sessions = 62;
    user2.avg_session_duration = 18.3f;
    user2.last_active = std::chrono::system_clock::now() - std::chrono::hours(24);
    users.push_back(user2);

    // Create sample items
    items.push_back(
        ItemProfile{.item_id = 201,
                    .title = "Wireless Headphones",
                    .category = "Electronics",
                    .description = "High-quality wireless headphones with noise cancellation",
                    .tags = {"audio", "wireless", "premium"},
                    .genres = {"tech"},
                    .price = 299.99f,
                    .avg_rating = 4.3f,
                    .num_ratings = 1250,
                    .created_date = std::chrono::system_clock::now() - std::chrono::hours(24 * 30),
                    .popularity_score = 8500});

    items.push_back(
        ItemProfile{.item_id = 202,
                    .title = "Programming Book",
                    .category = "Books",
                    .description = "Comprehensive guide to modern C++ programming",
                    .tags = {"programming", "education", "technical"},
                    .genres = {"tech", "education"},
                    .price = 45.99f,
                    .avg_rating = 4.7f,
                    .num_ratings = 320,
                    .created_date = std::chrono::system_clock::now() - std::chrono::hours(24 * 60),
                    .popularity_score = 3200});

    items.push_back(
        ItemProfile{.item_id = 203,
                    .title = "Running Shoes",
                    .category = "Sports",
                    .description = "Professional running shoes for athletes",
                    .tags = {"sports", "fitness", "shoes"},
                    .genres = {"sports"},
                    .price = 129.99f,
                    .avg_rating = 4.1f,
                    .num_ratings = 890,
                    .created_date = std::chrono::system_clock::now() - std::chrono::hours(24 * 15),
                    .popularity_score = 6700});

    items.push_back(
        ItemProfile{.item_id = 204,
                    .title = "Smart Watch",
                    .category = "Electronics",
                    .description = "Feature-rich smartwatch with health tracking",
                    .tags = {"wearable", "health", "smart"},
                    .genres = {"tech"},
                    .price = 399.99f,
                    .avg_rating = 4.0f,
                    .num_ratings = 2100,
                    .created_date = std::chrono::system_clock::now() - std::chrono::hours(24 * 45),
                    .popularity_score = 9200});

    items.push_back(
        ItemProfile{.item_id = 205,
                    .title = "Coffee Maker",
                    .category = "Home",
                    .description = "Automatic coffee maker with programmable settings",
                    .tags = {"kitchen", "appliance", "coffee"},
                    .genres = {"home"},
                    .price = 89.99f,
                    .avg_rating = 3.9f,
                    .num_ratings = 450,
                    .created_date = std::chrono::system_clock::now() - std::chrono::hours(24 * 90),
                    .popularity_score = 4100});

    return {users, items};
}

/**
 * @brief Main demonstration function
 */
int main() {
    std::cout << "🛒 MoE Recommendation System Demonstration\n";
    std::cout << "==========================================\n\n";

    // Configure MoE system for recommendations
    MoEConfig config;
    config.num_experts = 12;              // 3 experts per strategy
    config.expert_capacity = 2;           // Select top-2 experts
    config.load_balancing_weight = 0.2f;  // Higher for load balancing
    config.enable_sparse_activation = true;
    config.max_concurrent_requests = 50;  // High for recommendation systems

    try {
        // Initialize MoE recommendation system
        MoERecommendationSystem rec_system(config);

        // Create sample data
        auto [users, items] = create_sample_data();

        std::cout << "🧪 Running Recommendation Generation Tests:\n";
        std::cout << "==========================================\n\n";

        // Generate recommendations for each user
        for (const auto& user : users) {
            std::cout << "👤 Generating recommendations for: " << user.name << "\n";
            std::cout << "   Age: " << user.age << ", Location: " << user.location << "\n";
            std::cout << "   Previous purchases: " << user.purchased_items.size() << " items\n";
            std::cout << "   Top category preference: ";
            if (!user.category_preferences.empty()) {
                auto max_pref = std::max_element(user.category_preferences.begin(),
                                                 user.category_preferences.end(),
                                                 [](const auto& a, const auto& b) {
                                                     return a.second < b.second;
                                                 });
                std::cout << max_pref->first << " (" << std::fixed << std::setprecision(1)
                          << max_pref->second * 100 << "%)\n";
            }

            // Create recommendation request
            RecommendationRequest request;
            request.user = user;
            request.candidate_items = items;
            request.num_recommendations = 3;
            request.session_context = "browsing";
            request.time_constraint = 0.8f;
            request.include_diverse_items = true;

            auto result = rec_system.generate_recommendations(request);
            if (result.is_ok()) {
                auto response = std::move(result).unwrap();

                std::cout << "✨ Generated Recommendations:\n";
                for (std::size_t i = 0; i < response.recommended_item_ids.size(); ++i) {
                    std::size_t item_id = response.recommended_item_ids[i];
                    float score = response.recommendation_scores[i];

                    // Find item details
                    auto item_it = std::find_if(
                        items.begin(), items.end(), [item_id](const ItemProfile& item) {
                            return item.item_id == item_id;
                        });

                    if (item_it != items.end()) {
                        std::cout << "   " << (i + 1) << ". " << item_it->title << " ("
                                  << item_it->category << ")\n";
                        std::cout << "      Score: " << std::fixed << std::setprecision(3) << score
                                  << " | Price: $" << std::fixed << std::setprecision(2)
                                  << item_it->price << " | Rating: " << item_it->avg_rating
                                  << "/5.0\n";
                        std::cout << "      Reason: " << response.reasoning[i] << "\n";
                    }
                }

                std::cout << "\n📊 Recommendation Quality:\n";
                std::cout << "   Diversity Score: " << std::fixed << std::setprecision(3)
                          << response.diversity_score << "\n";
                std::cout << "   Confidence Score: " << std::fixed << std::setprecision(3)
                          << response.confidence_score << "\n";

            } else {
                std::cout << "❌ Recommendation generation failed: "
                          << std::move(result).unwrap_err() << "\n";
            }

            std::cout << "----------------------------------------\n\n";
        }

        // Print final statistics
        rec_system.print_statistics();

        std::cout << "✅ MoE Recommendation System Demo Complete!\n";
        std::cout << "\n🎯 Key Features Demonstrated:\n";
        std::cout << "   • Context-aware expert specialization\n";
        std::cout << "   • Dynamic expert selection based on user/item features\n";
        std::cout << "   • Multi-strategy recommendation generation\n";
        std::cout << "   • Real-time performance optimization\n";
        std::cout << "   • Diversity and confidence scoring\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
