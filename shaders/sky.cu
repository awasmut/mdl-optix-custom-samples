#include <optix.h>

#include <mi/neuraylib/target_code_types.h>


struct Params
{
    uchar4* image;

    unsigned int width;
    unsigned int height;

    float3 camera_position;

    float3 camera_u;
    float3 camera_v;
    float3 camera_w;
};

// MDL-generated environment callable (external, resolved at pipeline link time
// from the MDL module).
extern "C" __device__
void perez_sun_and_sky(void* result, void* state, void* resources, void* arg_block);

// OptiX direct-callable wrapper. The __direct_callable__ prefix is what OptiX
// uses to identify this as a direct callable semantic entry point.
extern "C" __device__
void __direct_callable__perez_sun_and_sky(void* result, void* state, void* resources, void* arg_block)
{
    perez_sun_and_sky(result, state, resources, arg_block);
}

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__
float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__
float3 mul3(const float s, const float3 v)
{
    return make_float3(s * v.x, s * v.y, s * v.z);
}

static __forceinline__ __device__ 
float3 normalize3(const float3 v)
{
    const float length_squared = v.x * v.x + v.y * v.y + v.z * v.z;
    const float inverse_length = rsqrtf(length_squared);

    return make_float3(v.x * inverse_length, v.y * inverse_length, v.z * inverse_length);
}

extern "C" __global__
void __raygen__sky()
{
    const uint3 launch_index = optixGetLaunchIndex();

    // Pixel center.
    const float2 pixel = make_float2(
        static_cast<float>(launch_index.x) + 0.5f,
        static_cast<float>(launch_index.y) + 0.5f
    );

    // Normalized [0,1] coordinates.
    const float2 uv = make_float2(
        pixel.x / static_cast<float>(params.width),
        pixel.y / static_cast<float>(params.height)
    );

    // Normalized image-plane coordinates [-1,+1].
    const float2 d = make_float2(
        uv.x * 2.0f - 1.0f,
        1.0f - uv.y * 2.0f
    );

    // Camera ray.
    const float3 direction = normalize3(add3(params.camera_w, add3(mul3(d.x, params.camera_u), mul3(d.y, params.camera_v))));

    // No geometry is needed for this sky-only renderer.
    // A null traversable causes OptiX to invoke the miss
    // program directly.
    optixTrace(
        0,
        params.camera_position,
        direction,
        0.0f,
        1.0e16f,
        0.0f, 
        OptixVisibilityMask(255),

        OPTIX_RAY_FLAG_DISABLE_ANYHIT |
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,

        0,      // SBT offset
        1,      // SBT stride
        0);     // miss SBT index
}


extern "C" __global__
void __miss__sky()
{
    // Direction of the environment ray.
    const float3 direction = optixGetWorldRayDirection();

    // Construct the MDL environment state.
    mi::neuraylib::Shading_state_environment state = {};
    state.direction = direction;

    // The MDL environment function writes its result through
    // the first argument.
    float3 radiance;

    // Invoke the MDL-generated environment callable.
    optixDirectCall<void>(0, &radiance, &state, nullptr, nullptr);

    // Simple exposure / tone mapping.
    const float exposure = 0.0002f;

    float3 c;
    c.x = 1.0f - expf(-radiance.x * exposure);
    c.y = 1.0f - expf(-radiance.y * exposure);
    c.z = 1.0f - expf(-radiance.z * exposure);

    // Gamma encode.
    c.x = powf(fmaxf(c.x, 0.0f), 1.0f / 2.2f);
    c.y = powf(fmaxf(c.y, 0.0f), 1.0f / 2.2f);
    c.z = powf(fmaxf(c.z, 0.0f), 1.0f / 2.2f);

    const unsigned int x = optixGetLaunchIndex().x;
    const unsigned int y = optixGetLaunchIndex().y;
    const unsigned int index = y * params.width + x;

    if (x == 512 && (y == 32 || y == 256 || y == 480))
    {
        printf("(x=%u, y=%u) dir=(%f, %f, %f)\n",
               x, y, direction.x, direction.y, direction.z);
    }

    params.image[index] = make_uchar4(
        static_cast<unsigned char>(255.0f * fminf(c.x, 1.0f)),
        static_cast<unsigned char>(255.0f * fminf(c.y, 1.0f)),
        static_cast<unsigned char>(255.0f * fminf(c.z, 1.0f)),
        255
    );
}