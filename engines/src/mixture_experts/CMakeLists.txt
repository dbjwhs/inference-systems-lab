# Mixture of Experts Implementation
# Phase 7C: Revolutionary Techniques - Production-ready MoE system

cmake_minimum_required(VERSION 3.16)

# === Core MoE Library ===
add_library(mixture_experts
    # Core engine implementation
    moe_engine.cpp
    moe_engine.hpp
    
    # Expert routing with learnable parameters  
    expert_router.hpp
    
    # Memory-efficient parameter management
    expert_parameters.hpp
    
    # SIMD-optimized sparse activation
    sparse_activation.hpp
    
    # Dynamic load balancing and dispatch
    load_balancer.hpp
    
    # Configuration and hyperparameter management
    moe_config.hpp
)

# === Compiler Requirements ===
target_compile_features(mixture_experts PUBLIC cxx_std_17)

# === Include Directories ===
target_include_directories(mixture_experts
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..  # For common/ includes
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# === Dependencies ===
# Link with existing common library for Result<T,E> patterns
target_link_libraries(mixture_experts
    PUBLIC
        common
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
)

# === Compiler Optimizations ===
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Enable aggressive optimizations for production performance
    target_compile_options(mixture_experts PRIVATE
        -O3                    # Maximum optimization
        -march=native          # Use all available CPU instructions
        -mtune=native          # Optimize for current CPU
        -ffast-math           # Fast math optimizations
        -DNDEBUG              # Disable debug assertions
    )
    
    # SIMD optimizations
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(mixture_experts PRIVATE
            -mavx2             # Enable AVX2 SIMD instructions
            -mfma              # Enable FMA instructions
        )
    endif()
endif()

# === Debug Configuration ===
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(mixture_experts PRIVATE
        MOE_DEBUG_ENABLED=1
        MOE_PERFORMANCE_MONITORING=1
    )
endif()

# === Feature Flags ===
option(MOE_ENABLE_SIMD "Enable SIMD optimizations for sparse activation" ON)
option(MOE_ENABLE_PROFILING "Enable detailed performance profiling" OFF)
option(MOE_ENABLE_VALIDATION "Enable runtime validation and checks" ON)

if(MOE_ENABLE_SIMD)
    target_compile_definitions(mixture_experts PUBLIC MOE_SIMD_ENABLED=1)
endif()

if(MOE_ENABLE_PROFILING)
    target_compile_definitions(mixture_experts PUBLIC MOE_PROFILING_ENABLED=1)
endif()

if(MOE_ENABLE_VALIDATION)
    target_compile_definitions(mixture_experts PUBLIC MOE_VALIDATION_ENABLED=1)
endif()

# === Static Analysis Integration ===
if(ENABLE_STATIC_ANALYSIS)
    find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
    if(CLANG_TIDY_COMMAND)
        set_target_properties(mixture_experts PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy"
        )
    endif()
endif()

# === Sanitizer Support ===
if(SANITIZER_TYPE)
    if(SANITIZER_TYPE STREQUAL "address" OR SANITIZER_TYPE STREQUAL "address+undefined")
        target_compile_options(mixture_experts PRIVATE -fsanitize=address)
        target_link_libraries(mixture_experts PRIVATE -fsanitize=address)
    endif()
    
    if(SANITIZER_TYPE STREQUAL "thread")
        target_compile_options(mixture_experts PRIVATE -fsanitize=thread)
        target_link_libraries(mixture_experts PRIVATE -fsanitize=thread)
    endif()
    
    if(SANITIZER_TYPE STREQUAL "undefined" OR SANITIZER_TYPE STREQUAL "address+undefined")
        target_compile_options(mixture_experts PRIVATE -fsanitize=undefined)
        target_link_libraries(mixture_experts PRIVATE -fsanitize=undefined)
    endif()
endif()

# === Installation ===
install(TARGETS mixture_experts
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    moe_engine.hpp
    expert_router.hpp  
    expert_parameters.hpp
    sparse_activation.hpp
    load_balancer.hpp
    moe_config.hpp
    DESTINATION include/engines/mixture_experts
)

# === Testing Suite ===
if(BUILD_TESTING)
    # MoE comprehensive tests
    add_executable(moe_tests
        tests/test_moe_engine.cpp
        tests/test_expert_router.cpp
        tests/test_load_balancer.cpp
        tests/test_moe_config.cpp
        tests/test_moe_integration.cpp
    )

    target_link_libraries(moe_tests
        mixture_experts
        GTest::gtest_main
    )

    # Add individual test registration for CTest integration
    add_test(NAME MoEEngineTests COMMAND moe_tests --gtest_filter="MoEEngineTest.*")
    add_test(NAME ExpertRouterTests COMMAND moe_tests --gtest_filter="ExpertRouterTest.*")
    add_test(NAME LoadBalancerTests COMMAND moe_tests --gtest_filter="LoadBalancerTest.*")
    add_test(NAME MoEConfigTests COMMAND moe_tests --gtest_filter="MoEConfigTest.*")
    add_test(NAME MoEIntegrationTests COMMAND moe_tests --gtest_filter="MoEIntegrationTest.*")
endif()

# === Benchmark Suite ===
if(BUILD_BENCHMARKS)
    add_executable(moe_benchmarks
        benchmarks/moe_benchmarks.cpp
    )

    target_link_libraries(moe_benchmarks
        mixture_experts
        benchmark::benchmark_main
    )

    # Set benchmark properties
    set_target_properties(moe_benchmarks PROPERTIES
        COMPILE_FLAGS "-O3"
    )
endif()

# === Documentation ===
set_target_properties(mixture_experts PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    DESCRIPTION "Production-ready Mixture of Experts system with sparse activation and dynamic dispatch"
    PUBLIC_HEADER "moe_engine.hpp;expert_router.hpp;expert_parameters.hpp;sparse_activation.hpp;load_balancer.hpp;moe_config.hpp"
)
