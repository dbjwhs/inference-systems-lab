# CompilerOptions.cmake
# Configures compiler-specific warning flags and optimization settings

function(configure_compiler_options)
    # Compiler-specific options for GNU/Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
            -Wall -Wextra -Wpedantic
            -Wcast-align -Wcast-qual -Wctor-dtor-privacy
            -Wdisabled-optimization -Winit-self
            -Wmissing-declarations -Wmissing-include-dirs
            -Wold-style-cast -Woverloaded-virtual -Wredundant-decls
            -Wshadow -Wsign-promo
            -Wstrict-overflow=5
        )
        
        # Debug vs Release configurations
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_compile_options(-g -O0 -fno-omit-frame-pointer)
            add_compile_definitions(INFERENCE_LAB_DEBUG)
        else()
            add_compile_options(-O3 -DNDEBUG)
        endif()
    endif()

    # MSVC-specific options
    if(MSVC)
        add_compile_options(/W4 /permissive-)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_compile_options(/Od /Zi)
        else()
            add_compile_options(/O2)
        endif()
    endif()
    
    message(STATUS "Compiler options configured for ${CMAKE_CXX_COMPILER_ID}")
endfunction()