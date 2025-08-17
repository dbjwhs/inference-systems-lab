# Documentation.cmake
# Configures documentation generation and code formatting tools

function(configure_documentation)
    # Documentation target (requires Doxygen)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        
        if(EXISTS ${DOXYGEN_IN})
            configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
            
            add_custom_target(docs
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen"
            )
            message(STATUS "Doxygen documentation target configured")
        else()
            message(STATUS "Doxygen found but Doxyfile.in not found")
        endif()
    else()
        message(STATUS "Doxygen not found, documentation target not available")
    endif()
endfunction()

function(configure_formatting)
    # Format target (requires clang-format)
    find_program(CLANG_FORMAT_EXE NAMES "clang-format")
    if(CLANG_FORMAT_EXE)
        file(GLOB_RECURSE ALL_SOURCE_FILES
            ${CMAKE_SOURCE_DIR}/**/src/*.cpp
            ${CMAKE_SOURCE_DIR}/**/src/*.hpp
            ${CMAKE_SOURCE_DIR}/**/tests/*.cpp
            ${CMAKE_SOURCE_DIR}/**/benchmarks/*.cpp
            ${CMAKE_SOURCE_DIR}/**/examples/*.cpp
        )
        
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
            COMMENT "Formatting source code with clang-format"
        )
        message(STATUS "Code formatting target configured")
    else()
        message(STATUS "clang-format not found, format target not available")
    endif()
endfunction()

# Helper function to display documentation status in configuration summary
function(display_documentation_status)
    if(DOXYGEN_FOUND)
        message(STATUS "  Doxygen: YES")
    else()
        message(STATUS "  Doxygen: NO")
    endif()
    
    find_program(CLANG_FORMAT_EXE NAMES "clang-format")
    if(CLANG_FORMAT_EXE)
        message(STATUS "  clang-format: YES")
    else()
        message(STATUS "  clang-format: NO")
    endif()
endfunction()