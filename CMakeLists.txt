cmake_minimum_required(VERSION 3.19)

project(neural_image LANGUAGES CXX)


set(CPM_DOWNLOAD_VERSION 0.38.2)

if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
    )
endif ()

include(${CPM_DOWNLOAD_LOCATION})


CPMAddPackage(
        GITLAB_REPOSITORY libeigen/eigen
        GIT_TAG 3.4.0
        DOWNLOAD_ONLY YES
)
CPMAddPackage(
        GITHUB_REPOSITORY nothings/stb
        GIT_TAG 5736b15
        DOWNLOAD_ONLY YES
)
CPMAddPackage(
        GITHUB_REPOSITORY muellan/clipp
        GIT_TAG v1.2.3
        DOWNLOAD_ONLY YES
)


add_executable(neural_image
        src/main.cpp
        src/network.cpp src/network.hpp
)
target_include_directories(neural_image SYSTEM PRIVATE
        ${eigen_SOURCE_DIR}
        ${stb_SOURCE_DIR}
        ${clipp_SOURCE_DIR}/include
)
target_compile_features(neural_image PRIVATE cxx_std_20)
target_compile_definitions(neural_image PRIVATE
        $<$<CONFIG:Debug>:EIGEN_RUNTIME_NO_MALLOC>
        EIGEN_NO_AUTOMATIC_RESIZING
)


set(CLANG_OPTIONS
        -march=native
        -ffast-math
        -Wfatal-errors
        -Wall
        -Wextra
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion)

set(GCC_OPTIONS
        ${CLANG_OPTIONS}
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wuseless-cast
        -Wswitch-enum)

if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(neural_image PRIVATE ${CLANG_OPTIONS})
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(neural_image PRIVATE ${GCC_OPTIONS})
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(neural_image PRIVATE /W4 /fp:fast)
else ()
    message(WARNING "No warnings set for compiler '${CMAKE_CXX_COMPILER_ID}'")
endif ()


message(CHECK_START "Checking for IPO support")
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported_result OUTPUT ipo_supported_output)
if (ipo_supported_result)
    message(CHECK_PASS "supported")
    set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION_MIN_SIZE_REL TRUE)
else ()
    message(CHECK_FAIL "not supported")
endif ()
