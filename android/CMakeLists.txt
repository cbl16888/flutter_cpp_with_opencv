cmake_minimum_required(VERSION 3.4.1)

include_directories(../include)
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/nativeLibs/${ANDROID_ABI}/libopencv_java3.so)
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")

find_library(log-lib log)

add_library(native_with_opencv SHARED ../ios/Classes/native_add.cpp)
target_link_libraries(native_with_opencv lib_opencv ${log-lib})