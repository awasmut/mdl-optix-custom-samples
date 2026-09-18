cmake_minimum_required(VERSION 3.25)

include(FetchContent)

project(mdl_optix9_sky LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(OPTIX_ROOT "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.1.0")
set(MDL_SDK_ROOT "C:\\ProgramData\\NVIDIA Corporation\\MDL-SDK-2026.0.0")

if(NOT DEFINED OPTIX_ROOT)
    message(FATAL_ERROR "Set OPTIX_ROOT to your OptiX 9.1 SDK")
endif()

if(NOT DEFINED MDL_SDK_ROOT)
    message(FATAL_ERROR "Set MDL_SDK_ROOT to your MDL SDK")
endif()

if(WIN32)
    set(MDL_SDK_DLL
        "${MDL_SDK_ROOT}/bin/libmdl_sdk.dll")
else()
    set(MDL_SDK_DLL
        "${MDL_SDK_ROOT}/lib/libmdl_sdk.so")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY  https://github.com/glfw/glfw.git
    GIT_TAG         3.5.1
    GIT_SHALLOW     TRUE
)
FetchContent_MakeAvailable(glfw)

set(SKY_CU ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sky.cu)
set(SKY_PTX ${CMAKE_CURRENT_BINARY_DIR}/sky.ptx)    
add_custom_command(
    OUTPUT ${SKY_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx
        --std=c++17
        -arch=compute_89
        -rdc=true
        -I${OPTIX_ROOT}/include
        -I${MDL_SDK_ROOT}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        -o ${SKY_PTX}
        ${SKY_CU}
    DEPENDS ${SKY_CU} ${CMAKE_CURRENT_SOURCE_DIR}/include/shared.hpp
    COMMENT "Compiling OptiX sky.cu -> sky.ptx"
    VERBATIM
)
add_custom_target(
    mdl_optix9_sky_ptx
    DEPENDS ${SKY_PTX}
)

add_executable(mdl_optix9_sky
    src/main.cpp
    src/mdl_loader.cpp
    src/mdl_sky.cpp
    external/glad/src/glad.c
)
add_dependencies(mdl_optix9_sky mdl_optix9_sky_ptx)

target_compile_definitions(
    mdl_optix9_sky 
    PRIVATE
        SKY_PTX_FILE="${SKY_PTX}"
)

add_custom_command(
    TARGET mdl_optix9_sky
    POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MDL_SDK_DLL}"
        "$<TARGET_FILE_DIR:mdl_optix9_sky>"
)

target_include_directories(mdl_optix9_sky PRIVATE
    ${OPTIX_ROOT}/include
    ${MDL_SDK_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    external/glad/include
)

target_link_libraries(mdl_optix9_sky PRIVATE
    CUDA::cudart
    OpenGL::GL
    glfw
)

set_target_properties(mdl_optix9_sky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_custom_command(
    TARGET mdl_optix9_sky POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:mdl_optix9_sky>/shaders
)
