cmake_minimum_required(VERSION 3.19)

set(WASI_SDK_PREFIX /home/tangshimei/wasi-sdk-14.0)
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()
if (NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT /usr/local/cuda)
endif ()

message("CUDAToolkit path = ${CUDAToolkit_ROOT}")
message("WASI-SDK path=${WASI_SDK_PREFIX}")

# set wasi-sdk toolchain
set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)

project(wasm-cuda-simple-examples)


include_directories(${CUDAToolkit_ROOT}/include)
include_directories(include)

add_compile_options(-fno-exceptions -O3 -Wextra -Werror)
add_link_options(-Wl,--allow-undefined,--export=main,--strip-all,--no-entry)


add_executable(2DConvolution.wasm src/2DConvolution.c)
add_executable(2mm.wasm src/2mm.c)
add_executable(adi.wasm src/adi.c)
add_executable(atax.wasm src/atax.c)
add_executable(bicg.wasm src/bicg.c)
add_executable(correlation.wasm src/correlation.c)
add_executable(doitgen.wasm src/doitgen.c)


