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

# Optimise the native code in every APK variant.
#
# The Android Gradle plugin compiles a debug variant's native code with
# CMAKE_BUILD_TYPE=Debug, which for ggml means -O0 and live assertions. For a
# model runtime that is not "a bit slower": it measured 0.3 tok/s on device.
# Debug info is kept, so this stays as debuggable as before — only the
# optimiser is turned back on.
# Appended as plain variables rather than forced into the cache: the NDK
# toolchain file sets these as normal variables at project() time, and a normal
# variable shadows the cache entry, so a FORCE here is silently ignored. This
# keeps the NDK's own debug flags and adds the optimiser back.
foreach(lang C CXX)
    set(CMAKE_${lang}_FLAGS_DEBUG          "${CMAKE_${lang}_FLAGS_DEBUG} -O3 -DNDEBUG")
    set(CMAKE_${lang}_FLAGS_RELWITHDEBINFO "${CMAKE_${lang}_FLAGS_RELWITHDEBINFO} -O3 -DNDEBUG")
endforeach()
message(STATUS "elias: C debug flags = ${CMAKE_C_FLAGS_DEBUG}")

# ARM CPU features.
#
# Cross-compiling leaves GGML_NATIVE off, and without GGML_CPU_ARM_ARCH ggml
# adds no -march at all — so the build targets baseline armv8-a and the
# quantised dot products take the scalar path. armv8.2-a+dotprod+fp16 is the
# floor for anything built after about 2018 and is where most of the speed is.
# Override with -PeliasArmArch=armv8-a if the app dies instantly with SIGILL on
# an older chip; see the README.
if(NOT DEFINED ELIAS_ARM_ARCH)
    set(ELIAS_ARM_ARCH "armv8.2-a+dotprod+fp16")
endif()
set(GGML_CPU_ARM_ARCH "${ELIAS_ARM_ARCH}" CACHE STRING "" FORCE)
message(STATUS "elias: targeting ${ELIAS_ARM_ARCH}")

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
