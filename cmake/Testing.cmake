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

# Coverage configuration (requires Debug build and GCC/Clang)
function(configure_coverage)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
        if(ENABLE_COVERAGE)
            add_compile_options(--coverage)
            add_link_options(--coverage)
            
            find_program(LCOV_EXE NAMES "lcov")
            find_program(GENHTML_EXE NAMES "genhtml")
            
            if(LCOV_EXE AND GENHTML_EXE)
                add_custom_target(coverage
                    COMMAND ${LCOV_EXE} --directory . --capture --output-file coverage.info
                    COMMAND ${LCOV_EXE} --remove coverage.info '/usr/*' --output-file coverage.info
                    COMMAND ${LCOV_EXE} --list coverage.info
                    COMMAND ${GENHTML_EXE} -o coverage coverage.info
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMENT "Generating coverage report"
                )
                message(STATUS "Coverage reporting enabled")
            else()
                message(WARNING "lcov and/or genhtml not found, coverage target not available")
            endif()
        endif()
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
