#pragma once

#include <optix.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <math.h>
#include <cuda_runtime.h>

struct Params
{
    uchar4* image;

    unsigned int width;
    unsigned int height;

    float3 camera_position;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;

    OptixTraversableHandle gas_handle;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CallableRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct HitGroupData
{
    float3 albedo;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    HitGroupData data;  
};