#include <optix.h>

#include <mi/neuraylib/target_code_types.h>

#include "shared.hpp"


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
float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
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

    const float2 uv = make_float2(
        pixel.x / static_cast<float>(params.width),
        pixel.y / static_cast<float>(params.height)
    );

    const float aspect = static_cast<float>(params.width) / static_cast<float>(params.height);
    // image-plane coordinates, aspect-corrected. 
    const float2 d = make_float2(
        (uv.x * 2.0f - 1.0f) * aspect,
        (1.0f - uv.y * 2.0f)
    );

    // Camera ray.
    const float3 forward = mul3(-1.0f, params.camera_w);
    const float3 direction = normalize3(add3(forward, add3(mul3(d.x, params.camera_u), mul3(d.y, params.camera_v))));

    unsigned int payload = 0;

    // No geometry is needed for this sky-only renderer.
    // A null traversable causes OptiX to invoke the miss
    // program directly.
    optixTrace(
        params.gas_handle,
        params.camera_position,
        direction,
        0.0f,
        1.0e16f,
        0.0f, 
        OptixVisibilityMask(255),

        OPTIX_RAY_FLAG_DISABLE_ANYHIT,

        0,      // SBT offset
        1,      // SBT stride
        0,      // miss SBT index

        payload);
}


extern "C" __global__
void __closesthit__sphere() 
{
    // Mark that we hit something.
    optixSetPayload_0(1);

    const HitGroupData* hitgroup_data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());

    // Geometric normal at the hit point.
    // The built-in sphere IS does not report a normal directly, but we can
    // reconstruct it from the hit position and the sphere center/radius.
    // For a unit sphere at the origin, normal = normalize(hit_position).
    // Here we need the sphere center and radius; pass them via launch params
    // or store them in the SBT record.
    const float3 hit_pos = add3(optixGetWorldRayOrigin(), mul3(optixGetRayTmax(), optixGetWorldRayDirection()));

    // For a sphere centered at C with radius R: normal = (hit_pos - C) / R
    // We'll store center and radius in the SBT record for generality.
    const float3 center = make_float3(0.0f, 0.0f, 5.0f);
    //const float radius = 1.0f;
    const float3 normal = normalize3(sub3(hit_pos, center));

    // Hardcoded light direction.
    const float3 light_dir = normalize3(make_float3(-1.0f, -1.0f, -1.0f));

    const float NoL = fmaxf(0.0f, normal.x * light_dir.x + normal.y * light_dir.y + normal.z * light_dir.z);

    float3 color = make_float3(
        hitgroup_data->albedo.x * NoL,
        hitgroup_data->albedo.y * NoL,
        hitgroup_data->albedo.z * NoL);

    // Tone map and gamma, same as the miss shader.
    const float exposure = 1.0f;
    //const float exposure = 0.0002f;
    color.x = 1.0f - expf(-color.x * exposure);
    color.y = 1.0f - expf(-color.y * exposure);
    color.z = 1.0f - expf(-color.z * exposure);
    color.x = powf(fmaxf(color.x, 0.0f), 1.0f / 2.2f);
    color.y = powf(fmaxf(color.y, 0.0f), 1.0f / 2.2f);
    color.z = powf(fmaxf(color.z, 0.0f), 1.0f / 2.2f);

    const unsigned int x = optixGetLaunchIndex().x;
    const unsigned int y = optixGetLaunchIndex().y;
    const unsigned int index = y * params.width + x;

    params.image[index] = make_uchar4(
        static_cast<unsigned char>(255.0f * fminf(color.x, 1.0f)),
        static_cast<unsigned char>(255.0f * fminf(color.y, 1.0f)),
        static_cast<unsigned char>(255.0f * fminf(color.z, 1.0f)),
        255);
    
    //params.image[index] = make_uchar4(255, 0, 0, 255);
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