cmake_minimum_required(VERSION 3.4.1)
project(tflite)

set (TFLITE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set (TFLITE_LIB_PATH "${TFLITE_PATH}/lib/${ANDROID_ABI}")
set (TFLITE_INCLUDE_PATH "${TFLITE_PATH}/include")
set (CUTEMODEL_PATH "${TFLITE_PATH}/cutemodel")

add_library(lib_tflite SHARED IMPORTED)
set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIB_PATH}/libtensorflowlite_c.so)

add_library(lib_tflite_gpu SHARED IMPORTED)
set_target_properties(lib_tflite_gpu PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIB_PATH}/libtensorflowlite_gpu_delegate.so)

add_library(lib_nnapi STATIC IMPORTED)
set_target_properties(lib_nnapi PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIB_PATH}/libnnapi_delegate.a)

add_library(lib_nnapi_impl STATIC IMPORTED)
set_target_properties(lib_nnapi_impl PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIB_PATH}/libnnapi_implementation.a)

add_library(lib_nnapi_util STATIC IMPORTED)
set_target_properties(lib_nnapi_util PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIB_PATH}/libnnapi_util.a)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

add_library(tflite SHARED
        ${CUTEMODEL_PATH}/CuteModel.cpp)


target_include_directories(tflite PUBLIC
        ${TFLITE_PATH}
        ${TFLITE_INCLUDE_PATH}
        ${TFLITE_INCLUDE_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include
        ${TFLITE_INCLUDE_PATH}/tensorflow/lite/tools/make/downloads/absl
        ${TFLITE_INCLUDE_PATH}/tensorflow/lite/tools/make/downloads/absl/absl
        )

target_link_libraries(tflite
        lib_tflite
        lib_tflite_gpu
        lib_nnapi
        lib_nnapi_impl
        lib_nnapi_util
        log
        )
