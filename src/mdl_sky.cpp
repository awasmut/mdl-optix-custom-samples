#include "mdl_sky.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <fstream>

#include <mi/base/handle.h>
#include <mi/base/types.h>

#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/iscope.h>

#include <mi/neuraylib/version.h>

#include <mi/neuraylib/target_code_types.h>

namespace
{

template <class T>
void check_interface(T* ptr, const char* name)
{
    if (!ptr) throw std::runtime_error(std::string("MDL SDK: failed to acquire ") + name);
}

void check_result(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

// Note: not used!
void dump_execution_errors(mi::neuraylib::IMdl_execution_context* context)
{
    if (!context) return;

    const mi::Size count = context->get_error_messages_count();
    for (mi::Size i = 0; i < count; ++i)
    {
        const mi::neuraylib::IMessage* message = context->get_error_message(i);

        std::cerr
            << "MDL error [" << (message ? message->get_code() : 0) << "]: "
            << (message ? message->get_string() : "<no message>") << '\n';
    }
}

} // namespace


namespace mdl_sky
{

MdlSkyResult compile_perez_sun_and_sky(
    mi::neuraylib::INeuray* neuray,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* context)
{
    using namespace mi::neuraylib;

    if (!neuray) throw std::runtime_error("INeuray is null");
    if (!transaction) throw std::runtime_error("MDL transaction is null");

    // Obtain the MDL backend API.
    mi::base::Handle<IMdl_backend_api> backend_api(neuray->get_api_component<IMdl_backend_api>());
    if (!backend_api) throw std::runtime_error("Could not obtain IMdl_backend_api");

    // Ask MDL for its CUDA PTX backend.
    mi::base::Handle<IMdl_backend> backend(backend_api->get_backend(IMdl_backend_api::MB_CUDA_PTX));
    if (!backend) throw std::runtime_error("Could not create MDL CUDA PTX backend");

    if(auto result = backend->set_option("sm_version", "86"); result != 0)
        throw std::runtime_error("Could not set MDL PTX backend sm_version.");

    // Obtain the definition of ::base::perez_sun_and_sky(...)
    // The base module is a built-in MDL module.
    mi::base::Handle<IMdl_factory> mdl_factory(neuray->get_api_component<IMdl_factory>());
    if (!mdl_factory) throw std::runtime_error("Could not obtain IMdl_factory");

    mi::base::Handle<const mi::IString> db_definition_name(mdl_factory->get_db_definition_name("::base::perez_sun_and_sky"));
    if (!db_definition_name) throw std::runtime_error("Could not determine DB name for ::base::perez_sun_and_sky");
    std::cout << "MDL DB definition name: " << db_definition_name->get_c_str() << "\n";
    std::cout.flush();

    // Note: using the db_definition_name to obtain the IFunction_definition pointer will not work.
    // Why?

    const std::string function_definition_name("mdl::base::perez_sun_and_sky(float,float,float3,color,float,bool,float,float,float)");
    mi::base::Handle<const IFunction_definition> definition(transaction->access<IFunction_definition>(function_definition_name.c_str()));
    if (!definition) throw std::runtime_error("Could not access perez_sun_and_sky definition");

    mi::Sint32 errors = 0;
    mi::base::Handle<IFunction_call> call(definition->create_function_call(nullptr, &errors));
    if (errors != 0) {
        std::cerr << "perez_sun_and_sky create_function_call reported " << errors << " error(s); arguments may be missing defaults.\n";
        std::cerr.flush();
    }
    if (!call || errors != 0) throw std::runtime_error("Could not create perez_sun_and_sky function call");

    const ITarget_code* target_code = backend->translate_environment(transaction, call.get(), "perez_sun_and_sky", context);
    if (!target_code) throw std::runtime_error("MDL failed to translate perez_sun_and_sky");

    // Report the generated callable interface.
    const mi::Size function_count = target_code->get_callable_function_count();
    assert(function_count == 1);
    std::cout << "MDL generated " << function_count << " callable function(s)\n";

    const char* ptx = target_code->get_code();
    const char* callable_name = target_code->get_callable_function(0);
    const char* callable_prototype = target_code->get_callable_function_prototype(0, ITarget_code::Prototype_language::SL_PTX);
    
    assert(ptx);
    assert(callable_name);
    assert(callable_prototype);

    std::cout << "MDL callable name: " << callable_name << '\n';
    std::cout << "MDL callable prototype:\n" << callable_prototype << '\n';
    {
        std::ofstream file("mdl_environment.ptx");
        file << ptx;
    }

    return {
        ptx ? std::string(ptx) : "",
        callable_name ? std::string(callable_name) : "",
        callable_prototype ? std::string(callable_prototype) : ""
    };
}

} // namespace mdl_sky
