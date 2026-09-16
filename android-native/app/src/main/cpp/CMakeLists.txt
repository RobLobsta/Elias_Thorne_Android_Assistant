cmake_minimum_required(VERSION 3.22)
project(elias C CXX)

# bitnet.cpp, vendored as a submodule. It is a llama.cpp fork carrying the I2_S
# ternary kernels; mainline llama.cpp cannot read BitNet's GGUF at all, because
# I2_S shares a type id with a quantisation mainline removed.
set(BITNET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bitnet" CACHE PATH "bitnet.cpp checkout")

if(NOT EXISTS "${BITNET_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "bitnet.cpp is missing from ${BITNET_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# Static everything. The NDK links shared libraries with --no-undefined, and
# ggml-base references the I2_S kernels that live over in ggml-cpu, so a shared
# split fails to link. One static blob inside libelias.so is what we want on
# Android anyway.
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
# arm64 uses the I2_S kernels directly; the TL1 lookup-table path needs a
# per-model codegen step and buys nothing for this export.
set(BITNET_ARM_TL1      OFF CACHE BOOL "" FORCE)
set(BITNET_X86_TL2      OFF CACHE BOOL "" FORCE)

add_subdirectory("${BITNET_DIR}" bitnet-build EXCLUDE_FROM_ALL)

add_library(elias SHARED
    elias_core.cpp
    elias_jni.cpp)

target_include_directories(elias PRIVATE
    "${BITNET_DIR}/3rdparty/llama.cpp/include"
    "${BITNET_DIR}/3rdparty/llama.cpp/ggml/include")

target_compile_features(elias PRIVATE cxx_std_17)
target_compile_options(elias PRIVATE -O3 -fvisibility=hidden)

target_link_libraries(elias
    llama
    ggml
    ggml-base
    ggml-cpu
    android
    log)
