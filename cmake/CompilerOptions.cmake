# CompilerOptions.cmake
# Configures compiler-specific warning flags and optimization settings

function(configure_compiler_options)
    # Compiler-specific options for GNU/Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Base optimization and debug flags apply globally
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_compile_options(-g -O0 -fno-omit-frame-pointer)
            add_compile_definitions(INFERENCE_LAB_DEBUG)
        else()
            add_compile_options(-O3 -DNDEBUG)
        endif()
    endif()
    
    # Create a function to apply strict warnings to our targets only
    # This avoids applying -Werror to external dependencies like GoogleTest/Benchmark
    function(apply_strict_warnings target_name)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target_name} PRIVATE
                -Wall -Wextra -Wpedantic
                -Wcast-align -Wcast-qual -Wctor-dtor-privacy
                -Wdisabled-optimization -Winit-self
                -Wmissing-declarations -Wmissing-include-dirs
                -Wold-style-cast -Woverloaded-virtual -Wredundant-decls
                -Wshadow -Wsign-promo
                -Wstrict-overflow=5
            )
        elseif(MSVC)
            target_compile_options(${target_name} PRIVATE /W4 /permissive-)
        endif()
    endfunction()

    # MSVC-specific global options
    if(MSVC)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_compile_options(/Od /Zi)
        else()
            add_compile_options(/O2)
        endif()
    endif()
    
    message(STATUS "Compiler options configured for ${CMAKE_CXX_COMPILER_ID}")
endfunction()

# Make apply_strict_warnings available globally after configure_compiler_options is called
macro(make_strict_warnings_available)
    function(apply_strict_warnings target_name)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target_name} PRIVATE
                -Wall -Wextra -Wpedantic
                -Wcast-align -Wcast-qual -Wctor-dtor-privacy
                -Wdisabled-optimization -Winit-self
                -Wmissing-declarations -Wmissing-include-dirs
                -Wold-style-cast -Woverloaded-virtual -Wredundant-decls
                -Wshadow -Wsign-promo
                -Wstrict-overflow=5
            )
        elseif(MSVC)
            target_compile_options(${target_name} PRIVATE /W4 /permissive-)
        endif()
    endfunction()
endmacro()
