# Pybind11 Configuration
# Handles finding or fetching pybind11 for Python bindings

option(BUILD_PYTHON_BINDINGS "Build Python bindings for C++ libraries" OFF)

if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Python bindings enabled - configuring pybind11")
    
    # First try to find pybind11 installed on system
    find_package(pybind11 QUIET)
    
    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_VERSION}")
    else()
        message(STATUS "pybind11 not found, fetching from GitHub")
        
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1  # Latest stable release
            GIT_SHALLOW    TRUE
        )
        
        FetchContent_MakeAvailable(pybind11)
        message(STATUS "Fetched pybind11 v2.11.1")
    endif()
    
    # Find Python interpreter and development components
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    
    message(STATUS "Python interpreter: ${Python3_EXECUTABLE}")
    message(STATUS "Python version: ${Python3_VERSION}")
    message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python libraries: ${Python3_LIBRARIES}")
    
    # Set up installation directory for Python module
    if(NOT DEFINED PYTHON_MODULE_INSTALL_DIR)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c 
                "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        
        if(PYTHON_SITE_PACKAGES)
            set(PYTHON_MODULE_INSTALL_DIR "${PYTHON_SITE_PACKAGES}" 
                CACHE PATH "Installation directory for Python modules")
        else()
            set(PYTHON_MODULE_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" 
                CACHE PATH "Installation directory for Python modules")
        endif()
    endif()
    
    message(STATUS "Python module install dir: ${PYTHON_MODULE_INSTALL_DIR}")
    
    # Helper function to create Python test targets
    function(add_python_test TARGET_NAME SCRIPT_PATH)
        if(BUILD_TESTING)
            add_test(
                NAME ${TARGET_NAME}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT_PATH}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            
            # Set Python path to find our built module
            set_tests_properties(${TARGET_NAME} PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/engines/src/python_bindings:$ENV{PYTHONPATH}"
            )
        endif()
    endfunction()
    
else()
    message(STATUS "Python bindings disabled (BUILD_PYTHON_BINDINGS=OFF)")
endif()