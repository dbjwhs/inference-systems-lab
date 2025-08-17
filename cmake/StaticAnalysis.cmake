# StaticAnalysis.cmake
# Configures static analysis tools (clang-tidy, cppcheck, etc.)

function(configure_static_analysis)
    # clang-tidy integration
    option(ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
    if(ENABLE_CLANG_TIDY)
        find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
        if(CLANG_TIDY_EXE)
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
            message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        else()
            message(WARNING "clang-tidy not found")
        endif()
    endif()
    
    # cppcheck integration (placeholder for future implementation)
    option(ENABLE_CPPCHECK "Enable cppcheck analysis" OFF)
    if(ENABLE_CPPCHECK)
        find_program(CPPCHECK_EXE NAMES "cppcheck")
        if(CPPCHECK_EXE)
            message(STATUS "cppcheck found: ${CPPCHECK_EXE}")
            # TODO: Add cppcheck integration when needed
        else()
            message(WARNING "cppcheck not found")
        endif()
    endif()
    
    message(STATUS "Static analysis tools configured")
endfunction()

# Helper function to display static analysis status in configuration summary
function(display_static_analysis_status)
    message(STATUS "  clang-tidy: ${ENABLE_CLANG_TIDY}")
    if(DEFINED ENABLE_CPPCHECK)
        message(STATUS "  cppcheck: ${ENABLE_CPPCHECK}")
    endif()
endfunction()