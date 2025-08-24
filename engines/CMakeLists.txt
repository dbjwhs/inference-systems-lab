# Inference engines library

# Find SQLite3 for model registry
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(SQLITE3 QUIET sqlite3)
endif()

if(NOT SQLITE3_FOUND)
    # Fallback to find_package
    find_package(SQLite3 QUIET)
    if(SQLite3_FOUND)
        set(SQLITE3_LIBRARIES ${SQLite3_LIBRARIES})
        set(SQLITE3_INCLUDE_DIRS ${SQLite3_INCLUDE_DIRS})
    else()
        # System search for SQLite3
        find_library(SQLITE3_LIBRARIES sqlite3)
        if(SQLITE3_LIBRARIES)
            message(STATUS "Found SQLite3: ${SQLITE3_LIBRARIES}")
        else()
            message(WARNING "SQLite3 not found - model registry will be disabled")
            set(SQLITE3_LIBRARIES "")
        endif()
    endif()
endif()

# Create engines library
add_library(engines STATIC
    src/placeholder.cpp
    src/inference_engine.cpp
    src/forward_chaining.cpp
    src/model_registry.cpp
)

target_include_directories(engines PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(engines
    common
    Threads::Threads
    ${SQLITE3_LIBRARIES}
)

# Add SQLite include directories if found
if(SQLITE3_INCLUDE_DIRS)
    target_include_directories(engines PRIVATE ${SQLITE3_INCLUDE_DIRS})
endif()

# Engines tests
add_executable(engines_tests
    tests/placeholder.cpp
)

target_link_libraries(engines_tests
    engines
    GTest::gtest_main
)

add_test(NAME EnginesTests COMMAND engines_tests)

# Engines benchmarks (using placeholder for now)
add_executable(engines_benchmarks
    benchmarks/placeholder.cpp
)

target_link_libraries(engines_benchmarks
    engines
    benchmark::benchmark_main
)

# Forward chaining demos have been replaced with simple_forward_chaining_demo

# Simple forward chaining demo
add_executable(simple_forward_chaining_demo
    examples/simple_forward_chaining_demo.cpp
)

target_link_libraries(simple_forward_chaining_demo
    engines
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(src/python_bindings)
endif()

