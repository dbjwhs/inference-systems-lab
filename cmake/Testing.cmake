# Testing.cmake
# Configures GoogleTest integration and CTest setup

function(configure_testing)
    # Enable testing framework and CTest integration
    include(CTest)
    enable_testing()
    
    # Google Test - prefer system-installed version for ABI compatibility
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "System GoogleTest not found, using FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(STATUS "Using system-installed GoogleTest: ${GTEST_VERSION}")
    endif()
    
    message(STATUS "Testing framework configured")
endfunction()

# Coverage configuration with enterprise-grade reporting
function(configure_coverage)
    option(ENABLE_COVERAGE "Enable coverage reporting (requires Debug build and GCC/Clang)" OFF)
    
    if(ENABLE_COVERAGE)
        if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(FATAL_ERROR "Coverage reporting requires Debug build. Use: cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON")
        endif()
        
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            message(FATAL_ERROR "Coverage reporting requires GCC or Clang compiler")
        endif()
        
        # Add coverage flags to all targets
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
        
        # Find coverage tools - prefer gcovr on macOS for better Clang compatibility
        find_program(LCOV_EXE NAMES "lcov")
        find_program(GENHTML_EXE NAMES "genhtml")
        find_program(GCOVR_EXE NAMES "gcovr")
        
        # Check for python gcovr module since it may not be in PATH
        if(NOT GCOVR_EXE)
            execute_process(
                COMMAND python3 -m gcovr --version
                RESULT_VARIABLE GCOVR_MODULE_RESULT
                OUTPUT_QUIET ERROR_QUIET
            )
            if(GCOVR_MODULE_RESULT EQUAL 0)
                set(GCOVR_EXE "python3;-m;gcovr")
                message(STATUS "Found gcovr as Python module")
            endif()
        endif()
        
        if(GCOVR_EXE)
            # gcovr-based coverage (better compatibility on macOS with Clang)
            add_custom_target(coverage
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                COMMAND ${GCOVR_EXE} --root ${CMAKE_SOURCE_DIR}
                    --filter "${CMAKE_SOURCE_DIR}/common/src/"
                    --exclude '.*tests/.*' 
                    --exclude '.*examples/.*'
                    --exclude '.*benchmarks/.*'
                    --merge-mode-functions=separate
                    --gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file
                    --html --html-details 
                    --output coverage.html
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating coverage report with gcovr..."
            )
            
            # Quick coverage summary target
            add_custom_target(coverage-summary
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                COMMAND ${GCOVR_EXE} --root ${CMAKE_SOURCE_DIR}
                    --filter "${CMAKE_SOURCE_DIR}/common/src/"
                    --exclude '.*tests/.*' 
                    --exclude '.*examples/.*'
                    --exclude '.*benchmarks/.*'
                    --merge-mode-functions=separate
                    --gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Quick coverage summary..."
            )
            
            message(STATUS "Coverage reporting enabled with gcovr - use 'make coverage' or 'make coverage-summary'")
        elseif(LCOV_EXE AND GENHTML_EXE)
            # Fallback to lcov if gcovr not available
            add_custom_target(coverage
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                COMMAND ${LCOV_EXE} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV_EXE} --remove coverage.info '/usr/*' '*/build/_deps/*' '*/tests/*' --output-file coverage_filtered.info
                COMMAND ${GENHTML_EXE} --output-directory coverage-report coverage_filtered.info
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating coverage report with lcov..."
            )
            message(STATUS "Coverage reporting enabled with lcov - use 'make coverage'")
        else()
            message(FATAL_ERROR "Coverage tools not found. Install with: sudo apt-get install lcov (or brew install lcov)")
        endif()
        
        # Coverage threshold checking
        set(COVERAGE_THRESHOLD 80 CACHE STRING "Minimum coverage percentage required")
        
        add_custom_target(coverage-enforce
            COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/check_coverage.py --threshold ${COVERAGE_THRESHOLD}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Enforcing ${COVERAGE_THRESHOLD}% coverage threshold..."
            DEPENDS coverage
        )
        
    endif()
endfunction()

# Helper function to display testing status in configuration summary
function(display_testing_status)
    if(GTest_FOUND)
        message(STATUS "  GoogleTest: System version")
    else()
        message(STATUS "  GoogleTest: FetchContent")
    endif()
    
    if(DEFINED ENABLE_COVERAGE)
        message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
    endif()
endfunction()
