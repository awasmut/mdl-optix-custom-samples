#include "mdl_sky.hpp"
#include "mdl_loader.hpp"
#include "shared.hpp"

#include <math.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/target_code_types.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t rc = (call);                                    \
        if (rc != cudaSuccess)                                      \
            throw std::runtime_error(                               \
                std::string("CUDA error: ") +                       \
                cudaGetErrorString(rc));                            \
    } while (0)

#define OPTIX_CHECK(call)                                           \
    do {                                                            \
        OptixResult rc = (call);                                    \
        if (rc != OPTIX_SUCCESS)                                    \
            throw std::runtime_error(                               \
                std::string("OptiX error: ") +                      \
                std::to_string(static_cast<int>(rc)));              \
    } while (0)


float3 add3(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static float3 sub3(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static float3 mul3(float s, float3 v)
{
    return make_float3(s * v.x, s * v.y, s * v.z);
}

static float3 normalize3(float3 v)
{
    const float length_squared = v.x * v.x + v.y * v.y + v.z * v.z;
    const float inverse_length = 1.0f / std::sqrtf(length_squared);

    return make_float3(v.x * inverse_length, v.y * inverse_length, v.z * inverse_length);
}

static float3 cross3(float3 a, float3 b) 
{
    return make_float3(
        a.y * b.z - a.z * b.y, // a2b3 - a3b2
        a.z * b.x - a.x * b.z, // a3b1 - a1b3
        a.x * b.y - a.y * b.x  // a1b2 - a2b1
    );
}

static std::string read_file(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open " + filename);

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}


static void save_ppm(const std::string& filename, const std::vector<uchar4>& pixels, unsigned int width, unsigned int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to create " + filename);
    file << "P6\n" << width << ' ' << height << "\n" << "255\n";

    for (const uchar4& p : pixels)
    {
        file.put(static_cast<char>(p.x));
        file.put(static_cast<char>(p.y));
        file.put(static_cast<char>(p.z));
    }
}

static OptixModule create_module(
    OptixDeviceContext optix_context,
    const std::string& ptx,
    const char* name,
    const OptixPipelineCompileOptions& pipeline_options,
    const OptixModuleCompileOptions& module_options)
{
    OptixModule module = nullptr;

    char log[8192];
    size_t log_size = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        optix_context,
        &module_options,
        &pipeline_options,
           ptx.data(),
        ptx.size(),
        log,
        &log_size,
        &module)
    );

    if (log_size > 1) std::cerr << name << " module log:\n" << log << '\n';

    return module;
}


int main(int argc, char** argv)
{
    try
    {
        // Configuration
        constexpr unsigned int width  = 1024;
        constexpr unsigned int height = 512;
        
        // CUDA
        CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp device_properties {};
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));
        std::cout << "GPU: " << device_properties.name << '\n';

        // OptiX
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions context_options = {};
        context_options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void*) {
                std::cerr << "[OptiX " << level << "] " << tag << ": " << message << '\n';
        };
        context_options.logCallbackLevel = 4;
        OptixDeviceContext optix_context = nullptr;
        OPTIX_CHECK(optixDeviceContextCreate(nullptr, &context_options, &optix_context));

        // MDL
        mi::base::Handle<mi::neuraylib::INeuray> neuray(mdl_loader::load());
        if (!neuray.is_valid_interface()) throw std::runtime_error("Could not load MDL SDK.");
        if (neuray->start(true) != 0) throw std::runtime_error("Could not start MDL SDK.");

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(neuray->get_api_component<mi::neuraylib::IMdl_factory>());
        if (!mdl_factory) throw std::runtime_error("Could not obtain IMdl_factory.");

        mi::base::Handle<mi::neuraylib::IDatabase> database(neuray->get_api_component<mi::neuraylib::IDatabase>());
        if (!database) throw std::runtime_error("Could not get MDL database.");

        mi::base::Handle<mi::neuraylib::IScope> global_scope(database->get_global_scope());
        if (!global_scope) throw std::runtime_error("Could not obtain MDL global scope.");
            
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(global_scope->create_transaction());
        if (!transaction) throw std::runtime_error("Could not create MDL transaction.");

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> mdl_context(mdl_factory->create_execution_context());
        if (!mdl_context) throw std::runtime_error("Could not create MDL execution context.");
            
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp(neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        if (!mdl_impexp) throw std::runtime_error("Could not obtain IMdl_impexp_api.");
                
        const mi::Sint32 load_result = mdl_impexp->load_module(transaction.get(), "::base", mdl_context.get());
        if (load_result != 0) throw std::runtime_error("Could not load MDL module ::base.");
        std::cout << "load_module(::base) returned " << load_result << "\n";
        std::cout.flush();

        // Compile the environment.
        // Keep the transaction alive while we consume target_code.
        const auto mdl_result = mdl_sky::compile_perez_sun_and_sky(neuray.get(), transaction.get(), mdl_context.get());
        transaction->commit();
        std::cout << "MDL environment compilation succeeded.\n";

        // ---------------------------------------------------------------
        // OptiX pipeline configuration
        // ---------------------------------------------------------------
        OptixPipelineCompileOptions pipeline_options = {};
        pipeline_options.usesMotionBlur = false;
        pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_options.numPayloadValues = 1;
        pipeline_options.numAttributeValues = 0; // built-in sphere reports no user attributes
        pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_options.pipelineLaunchParamsVariableName = "params";
        pipeline_options.pipelineLaunchParamsSizeInBytes = sizeof(Params);
        pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        pipeline_options.allowOpacityMicromaps = false;
        pipeline_options.allowClusteredGeometry = false;
        // ---------------------------------------------------------------
        // OptiX pipeline configuration
        // ---------------------------------------------------------------
        OptixModuleCompileOptions module_options = {};
        module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

        // ---------------------------------------------------------------
        // Create application module.
        // ---------------------------------------------------------------
#ifndef SKY_PTX_FILE
#error "SKY_PTX_FILE was not defined by CMake"
#endif
        const std::string sky_ptx = read_file(std::string(SKY_PTX_FILE));
        OptixModule app_module = create_module(optix_context, sky_ptx, "application", pipeline_options, module_options);

        // ---------------------------------------------------------------
        // Create MDL module.
        //
        // IMPORTANT:
        //
        // The MDL-generated PTX is itself a perfectly valid OptiX module.
        // We don't concatenate it with our application PTX.
        //
        // The generated callable becomes an OptiX direct-callable program.
        // ---------------------------------------------------------------
        std::string mdl_ptx_with_entry = mdl_result.ptx;
        mdl_ptx_with_entry += 
            "\n"
            ".visible .entry __raygen__mdl_dummy()\n"
            "{\n"
            "    ret;\n"
            "}\n";        

        OptixModule mdl_module = create_module(optix_context, mdl_ptx_with_entry, "MDL", pipeline_options, module_options);


        // ---------------------------------------------------------------
        // Sphere data
        // ---------------------------------------------------------------
        float3 sphere_center = make_float3(0.0f, 0.0f, 5.0f);
        float sphere_radius = 1.0f;

        CUdeviceptr d_sphere_centers = 0;
        CUdeviceptr d_sphere_radii = 0;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphere_centers), sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sphere_centers), &sphere_center, sizeof(float3), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphere_radii), sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sphere_radii), &sphere_radius, sizeof(float), cudaMemcpyHostToDevice));

        // ---------------------------------------------------------------
        // Sphere build input
        // ---------------------------------------------------------------
        unsigned int sphere_build_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

        OptixBuildInput build_input = {};
        build_input.type = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_SPHERES;
        build_input.sphereArray.vertexBuffers = &d_sphere_centers;
        build_input.sphereArray.vertexStrideInBytes = sizeof(float3);
        build_input.sphereArray.numVertices = 1;
        build_input.sphereArray.radiusBuffers = &d_sphere_radii;
        build_input.sphereArray.radiusStrideInBytes = sizeof(float);
        build_input.sphereArray.singleRadius = false;
        build_input.sphereArray.flags = sphere_build_flags;
        build_input.sphereArray.numSbtRecords = 1;
        build_input.sphereArray.sbtIndexOffsetBuffer = 0;
        build_input.sphereArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.sphereArray.sbtIndexOffsetStrideInBytes = 0;
        build_input.sphereArray.primitiveIndexOffset = 0;
        // ---------------------------------------------------------------
        // Sphere build
        // ---------------------------------------------------------------
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes buffer_sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &accel_options, &build_input, 1, &buffer_sizes));

        CUdeviceptr d_temp_buffer = 0;
        CUdeviceptr d_gas_output = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output), buffer_sizes.outputSizeInBytes));

        OptixTraversableHandle gas_handle = 0;
        OPTIX_CHECK(optixAccelBuild(
            optix_context,
            /*stream*/ 0,
            &accel_options,
            &build_input,
            1,
            d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            d_gas_output,
            buffer_sizes.outputSizeInBytes,
            &gas_handle,
            /*emittedProperties*/ nullptr,
            /*numEmittedPropeties*/ 0
        ));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
        // ---------------------------------------------------------------
        // OptiX sphere (IS) module
        // ---------------------------------------------------------------
        OptixModule sphere_is_module = nullptr;
        OptixBuiltinISOptions builtin_is_options = {};
        builtin_is_options.builtinISModuleType = OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE;
        builtin_is_options.usesMotionBlur = false;
        builtin_is_options.buildFlags = 0;
        builtin_is_options.curveEndcapFlags = 0;

        OPTIX_CHECK(optixBuiltinISModuleGet(
            optix_context,
            &module_options,
            &pipeline_options,
            &builtin_is_options,
            &sphere_is_module
        ));


        // ---------------------------------------------------------------
        // Program groups
        // ---------------------------------------------------------------
        char log[8192];
        size_t log_size = sizeof(log);
        OptixProgramGroupOptions pg_options = {};

        // Ray generation
        OptixProgramGroup raygen_pg = nullptr;
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = app_module;
        raygen_desc.raygen.entryFunctionName = "__raygen__sky";

        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &raygen_desc, 1, &pg_options, log, &log_size, &raygen_pg));

        // Miss
        OptixProgramGroup miss_pg = nullptr;
        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_desc.miss.module = app_module;
        miss_desc.miss.entryFunctionName = "__miss__sky";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &miss_desc, 1, &pg_options, log, &log_size, &miss_pg));

        // MDL direct callable
        OptixProgramGroup mdl_callable_pg = nullptr;
        OptixProgramGroupDesc callable_desc = {};
        callable_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_desc.callables.moduleDC = app_module;
        callable_desc.callables.entryFunctionNameDC = "__direct_callable__perez_sun_and_sky";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &callable_desc, 1, &pg_options, log, &log_size, &mdl_callable_pg));
        
        // Dummy raygen program group. Never scheduled or launched. Its only purpose
        // is to force the pipeline linker to include mdl_module, which provides the
        // definition of perez_sun_and_sky that the wrapper in app_module references.
        OptixProgramGroup mdl_dummy_pg = nullptr;
        OptixProgramGroupDesc mdl_dummy_desc = {};
        mdl_dummy_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        mdl_dummy_desc.raygen.module = mdl_module;
        mdl_dummy_desc.raygen.entryFunctionName = "__raygen__mdl_dummy";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &mdl_dummy_desc, 1, &pg_options, log, &log_size, &mdl_dummy_pg));   

        // sphere hit
        OptixProgramGroup hit_pg = nullptr;
        OptixProgramGroupDesc hit_desc = {};
        hit_desc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_desc.flags = 0;
        hit_desc.hitgroup.moduleCH = app_module;
        hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
        hit_desc.hitgroup.moduleAH = nullptr;
        hit_desc.hitgroup.entryFunctionNameAH = nullptr;
        hit_desc.hitgroup.moduleIS = sphere_is_module;
        hit_desc.hitgroup.entryFunctionNameIS = nullptr;

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &hit_desc, 1, &pg_options, log, &log_size, &hit_pg));   


        if (log_size > 1) std::cerr << "MDL callable program group log:\n" << log << '\n';
        // ---------------------------------------------------------------
        // Pipeline
        // ---------------------------------------------------------------
        std::vector<OptixProgramGroup> program_groups = {
            raygen_pg,
            miss_pg,
            mdl_callable_pg,
            mdl_dummy_pg,
            hit_pg
        };

        OptixPipeline pipeline = nullptr;
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = 1;

        log_size = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            optix_context,
            &pipeline_options,
            &pipeline_link_options,
            program_groups.data(),
            static_cast<unsigned int>(program_groups.size()),
            log,
            &log_size,
            &pipeline)
        );

        // ---------------------------------------------------------------
        // Stack size
        // ---------------------------------------------------------------
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 2048, 2048, 2048, 1));
        // ---------------------------------------------------------------
        // SBT
        // ---------------------------------------------------------------
        RaygenRecord raygen_record {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &raygen_record));

        MissRecord miss_record {};
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &miss_record));

        CallableRecord callable_record {};
        OPTIX_CHECK(optixSbtRecordPackHeader(mdl_callable_pg, &callable_record));

        HitGroupRecord hit_record {};
        OPTIX_CHECK(optixSbtRecordPackHeader(hit_pg, &hit_record));
        hit_record.data.albedo = make_float3(0.8f, 0.4f, 0.2f);

        CUdeviceptr d_raygen_record     = 0;
        CUdeviceptr d_miss_record       = 0;
        CUdeviceptr d_callable_record   = 0;
        CUdeviceptr d_hit_record        = 0;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(raygen_record)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(miss_record)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callable_record), sizeof(callable_record)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(hit_record)));

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof(raygen_record), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &miss_record, sizeof(miss_record), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_callable_record), &callable_record, sizeof(callable_record), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit_record), &hit_record, sizeof(hit_record), cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord                    = d_raygen_record;
        sbt.missRecordBase                  = d_miss_record;
        sbt.missRecordStrideInBytes         = sizeof(MissRecord);
        sbt.missRecordCount                 = 1;
        sbt.callablesRecordBase             = d_callable_record;
        sbt.callablesRecordStrideInBytes    = sizeof(CallableRecord);
        sbt.callablesRecordCount            = 1;
        sbt.hitgroupRecordBase              = d_hit_record;
        sbt.hitgroupRecordStrideInBytes     = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount             = 1;

        // ---------------------------------------------------------------
        // Output
        // ---------------------------------------------------------------
        std::vector<uchar4> host_pixels(width * height);
        uchar4* device_pixels = nullptr;
        CUDA_CHECK(cudaMalloc(&device_pixels, host_pixels.size() * sizeof(uchar4)));

        // ---------------------------------------------------------------
        // Camera
        // ---------------------------------------------------------------
        const float3 eye    = make_float3(0.0f, 0.5f, 0.0f);
        const float3 target = make_float3(0.0f, 0.0f, 5.0f);
        const float3 up     = make_float3(0.0f, 1.0f, 0.0f);

    
        const float3 w = normalize3(sub3(eye, target));  // backward
        const float3 u = normalize3(cross3(up, w));      // right
        const float3 v = cross3(w, u);                   // up

        Params params {};
        params.image = device_pixels;
        params.width = width;
        params.height = height;
        params.camera_position = eye;
        params.camera_u = u;
        params.camera_v = v;
        params.camera_w = w;
        params.gas_handle = gas_handle;

        // ---------------------------------------------------------------
        // Launch
        // ---------------------------------------------------------------
        CUdeviceptr d_params = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height,1));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(host_pixels.data(), device_pixels, host_pixels.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));

        save_ppm("perez_sun_and_sky.ppm", host_pixels, width, height);
        std::cout << "Wrote perez_sun_and_sky.ppm\n";

        // Cleanup
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
        CUDA_CHECK(cudaFree(device_pixels));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_callable_record)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_miss_record)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_raygen_record)));
        CUDA_CHECK(cudaStreamDestroy(stream));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(mdl_callable_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(miss_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(mdl_dummy_pg));
        OPTIX_CHECK(optixModuleDestroy(mdl_module));
        OPTIX_CHECK(optixModuleDestroy(app_module));
        OPTIX_CHECK(optixDeviceContextDestroy(optix_context));

        if (neuray->shutdown(true) != 0) throw std::runtime_error("Could not shut down MDL SDK.");
        neuray = nullptr;

        if (!mdl_loader::unload()) throw std::runtime_error("Could not unload MDL SDK.");
                
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nFATAL: " << e.what() << '\n';
        return 1;
    }
}
