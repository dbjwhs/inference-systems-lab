# PackageConfig.cmake
# Configures external package dependencies and FetchContent

function(configure_packages)
    # Find required packages
    find_package(Threads REQUIRED)
    message(STATUS "Found Threads: ${CMAKE_THREAD_LIBS_INIT}")
    
    # Optional packages for enhanced development
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        message(STATUS "Found PkgConfig")
    endif()
    
    message(STATUS "Package dependencies configured")
endfunction()

# Helper function to display package status in configuration summary
function(display_package_status)
    message(STATUS "  Threads: ${CMAKE_THREAD_LIBS_INIT}")
    if(PkgConfig_FOUND)
        message(STATUS "  PkgConfig: YES")
    else()
        message(STATUS "  PkgConfig: NO")
    endif()
endfunction()