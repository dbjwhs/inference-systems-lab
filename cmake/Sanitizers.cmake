# Sanitizers.cmake
# Configures runtime sanitizers for development and testing

function(configure_sanitizers)
    # Sanitizer support only available for GNU/Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Sanitizer build configurations
        set(SANITIZER_TYPE "none" CACHE STRING "Sanitizer to use (none, address, thread, memory, undefined, address+undefined)")
        set_property(CACHE SANITIZER_TYPE PROPERTY STRINGS "none;address;thread;memory;undefined;address+undefined")
        
        # Individual sanitizer options for advanced users
        option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
        option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
        option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
        option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
        
        # Apply sanitizers based on SANITIZER_TYPE
        if(SANITIZER_TYPE STREQUAL "address")
            set(ENABLE_ASAN ON)
            message(STATUS "Enabled AddressSanitizer")
        elseif(SANITIZER_TYPE STREQUAL "thread")
            set(ENABLE_TSAN ON)
            message(STATUS "Enabled ThreadSanitizer")
        elseif(SANITIZER_TYPE STREQUAL "memory")
            set(ENABLE_MSAN ON)
            message(STATUS "Enabled MemorySanitizer")
        elseif(SANITIZER_TYPE STREQUAL "undefined")
            set(ENABLE_UBSAN ON)
            message(STATUS "Enabled UndefinedBehaviorSanitizer")
        elseif(SANITIZER_TYPE STREQUAL "address+undefined")
            set(ENABLE_ASAN ON)
            set(ENABLE_UBSAN ON)
            message(STATUS "Enabled AddressSanitizer + UndefinedBehaviorSanitizer")
        endif()
        
        # Apply sanitizer flags
        set(SANITIZER_FLAGS "")
        set(SANITIZER_LINK_FLAGS "")
        
        if(ENABLE_ASAN)
            list(APPEND SANITIZER_FLAGS "-fsanitize=address" "-fno-omit-frame-pointer")
            list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=address")
            # Enable better stack traces
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                list(APPEND SANITIZER_FLAGS "-O1" "-g")
            endif()
        endif()
        
        if(ENABLE_TSAN)
            # ThreadSanitizer is incompatible with AddressSanitizer
            if(ENABLE_ASAN)
                message(FATAL_ERROR "ThreadSanitizer is incompatible with AddressSanitizer")
            endif()
            list(APPEND SANITIZER_FLAGS "-fsanitize=thread")
            list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=thread")
        endif()
        
        if(ENABLE_MSAN)
            # MemorySanitizer is incompatible with AddressSanitizer
            if(ENABLE_ASAN)
                message(FATAL_ERROR "MemorySanitizer is incompatible with AddressSanitizer")
            endif()
            list(APPEND SANITIZER_FLAGS "-fsanitize=memory" "-fno-omit-frame-pointer")
            list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=memory")
        endif()
        
        if(ENABLE_UBSAN)
            list(APPEND SANITIZER_FLAGS "-fsanitize=undefined")
            list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=undefined")
            # Additional UBSan checks
            list(APPEND SANITIZER_FLAGS 
                "-fsanitize=signed-integer-overflow"
                "-fsanitize=null"
                "-fsanitize=bounds"
                "-fsanitize=alignment"
                "-fsanitize=object-size"
                "-fsanitize=vptr"
            )
        endif()
        
        # Apply flags if any sanitizers are enabled
        if(SANITIZER_FLAGS)
            add_compile_options(${SANITIZER_FLAGS})
            add_link_options(${SANITIZER_LINK_FLAGS})
            
            # Set environment variables for better output
            # Note: detect_leaks is only supported on Linux x86_64/aarch64, not on macOS
            # Note: detect_container_overflow=0 disables false positives in mixed instrumentation scenarios
            # See: https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow
            # See: docs/ADDRESSSANITIZER_NOTES.md
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                set(ENV{ASAN_OPTIONS} "abort_on_error=1:fast_unwind_on_malloc=0:detect_leaks=1:detect_container_overflow=0")
                message(STATUS "LeakSanitizer enabled (Linux platform)")
            else()
                set(ENV{ASAN_OPTIONS} "abort_on_error=1:fast_unwind_on_malloc=0:detect_leaks=0:detect_container_overflow=0")
                message(STATUS "LeakSanitizer disabled (platform ${CMAKE_SYSTEM_NAME} not supported)")
            endif()
            message(STATUS "Container overflow detection disabled (prevents false positives)")
            set(ENV{UBSAN_OPTIONS} "abort_on_error=1:print_stacktrace=1")
            
            message(STATUS "Sanitizer flags: ${SANITIZER_FLAGS}")
            message(STATUS "Sanitizer link flags: ${SANITIZER_LINK_FLAGS}")
        endif()
    else()
        if(NOT SANITIZER_TYPE STREQUAL "none")
            message(WARNING "Sanitizers are only supported with GCC or Clang compilers")
        endif()
    endif()
endfunction()

# Helper function to display sanitizer status in configuration summary
function(display_sanitizer_status)
    if(DEFINED SANITIZER_TYPE)
        message(STATUS "  Sanitizer: ${SANITIZER_TYPE}")
        if(DEFINED SANITIZER_FLAGS AND SANITIZER_FLAGS)
            message(STATUS "    Address: ${ENABLE_ASAN}")
            message(STATUS "    Thread: ${ENABLE_TSAN}")
            message(STATUS "    Memory: ${ENABLE_MSAN}")
            message(STATUS "    UndefinedBehavior: ${ENABLE_UBSAN}")
        endif()
    endif()
endfunction()
