#include "mdl_sky.hpp"
#include "mdl_loader.hpp"

#include <cuda_runtime.h>

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

static OptixModule create_module(OptixDeviceContext optix_context, const std::string& ptx, const char* name, const OptixPipelineCompileOptions& pipeline_options)
{
    OptixModule module = nullptr;
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

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
        pipeline_options.numPayloadValues = 0;
        pipeline_options.numAttributeValues = 0;
        pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_options.pipelineLaunchParamsVariableName = "params";

        // ---------------------------------------------------------------
        // Create application module.
        // ---------------------------------------------------------------
#ifndef SKY_PTX_FILE
#error "SKY_PTX_FILE was not defined by CMake"
#endif
        const std::string sky_ptx = read_file(std::string(SKY_PTX_FILE));
        OptixModule app_module = create_module(optix_context, sky_ptx, "application", pipeline_options);

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

        OptixModule mdl_module = create_module(optix_context, mdl_ptx_with_entry, "MDL", pipeline_options);

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

        if (log_size > 1) std::cerr << "MDL callable program group log:\n" << log << '\n';
        // ---------------------------------------------------------------
        // Pipeline
        // ---------------------------------------------------------------
        std::vector<OptixProgramGroup> program_groups = {
            raygen_pg,
            miss_pg,
            mdl_callable_pg,
            mdl_dummy_pg
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

        CUdeviceptr d_raygen_record = 0;
        CUdeviceptr d_miss_record = 0;
        CUdeviceptr d_callable_record = 0;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(raygen_record)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(miss_record)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callable_record), sizeof(callable_record)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof(raygen_record), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &miss_record, sizeof(miss_record),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_callable_record), &callable_record, sizeof(callable_record), cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_raygen_record;
        sbt.missRecordBase = d_miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        sbt.callablesRecordBase =d_callable_record;
        sbt.callablesRecordStrideInBytes =sizeof(CallableRecord);
        sbt.callablesRecordCount = 1;

        // ---------------------------------------------------------------
        // Output
        // ---------------------------------------------------------------
        std::vector<uchar4> host_pixels(width * height);
        uchar4* device_pixels = nullptr;
        CUDA_CHECK(cudaMalloc(&device_pixels, host_pixels.size() * sizeof(uchar4)));

        // ---------------------------------------------------------------
        // Camera
        // ---------------------------------------------------------------
        const float3 camera_position = make_float3(0.0f, 0.0f, 0.0f);

        // Looking down +Z.
        // X = horizontal
        // Y = vertical
        // Z = forward
        const float3 camera_u = make_float3(1.0f, 0.0f, 0.0f);
        const float3 camera_v = make_float3(0.0f, 1.0f, 0.0f);
        const float3 camera_w = make_float3(0.0f, 0.0f, 1.0f);

        Params params {};
        params.image = device_pixels;
        params.width = width;
        params.height = height;
        params.camera_position = camera_position;
        params.camera_u = camera_u;
        params.camera_v = camera_v;
        params.camera_w = camera_w;

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
