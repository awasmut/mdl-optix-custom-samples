#include "mdl_loader.hpp"

#include <cstdio>
#include <string>

#ifdef MI_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#endif


namespace
{

#ifdef MI_PLATFORM_WINDOWS
HMODULE g_library = nullptr;
#else
void* g_library = nullptr;
#endif

using FactoryFunction = mi::neuraylib::INeuray* (*)(const mi::base::Uuid&);

} // namespace


namespace mdl_loader
{

mi::neuraylib::INeuray* load(const char* filename)
{
    if (g_library)
    {
        std::fprintf(stderr, "MDL SDK is already loaded.\n");
        return nullptr;
    }

    // NVIDIA's current SDK uses:
    //     libmdl_sdk.so
    // or
    //     mdl_sdk.dll
    //
    // MI_BASE_DLL_FILE_EXT supplies the platform-specific
    // extension.
    std::string library_name;

    if (filename)
    {
        library_name = filename;
    }
    else
    {
#ifdef _WIN32
    library_name = "libmdl_sdk.dll";
#else
    library_name = "libmdl_sdk.so";
#endif
    }

#ifdef MI_PLATFORM_WINDOWS

    HMODULE library = LoadLibraryA(library_name.c_str());

    if (!library)
    {
        std::fprintf(stderr, "Unable to load MDL SDK: %s\n", library_name.c_str());
        return nullptr;
    }

    FARPROC symbol = GetProcAddress(library, "mi_factory");
    if (!symbol)
    {
        std::fprintf(stderr, "MDL SDK does not export mi_factory().\n");
        FreeLibrary(library);
        return nullptr;
    }
    g_library = library;
#else

    void* library = dlopen(library_name.c_str(), RTLD_LAZY | RTLD_LOCAL);

    if (!library)
    {
        std::fprintf(stderr, "Unable to load MDL SDK: %s\n" "  %s\n", library_name.c_str(), dlerror());
        return nullptr;
    }

    void* symbol = dlsym(library, "mi_factory");

    if (!symbol)
    {
        std::fprintf(stderr, "MDL SDK does not export mi_factory():\n" "  %s\n", dlerror());
        dlclose(library);
        return nullptr;
    }

    g_library = library;
#endif

    // mi_factory() is the MDL SDK's public entry point.
    // Use the SDK helper rather than casting the symbol directly.
#ifdef MI_PLATFORM_WINDOWS
    auto* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(reinterpret_cast<void*>(symbol));
#else
    auto* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
#endif

    if (neuray) return neuray;

    // If INeuray couldn't be obtained, try IVersion so that
    // an ABI/header mismatch can be diagnosed.
#ifdef MI_PLATFORM_WINDOWS

    auto* version = mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(reinterpret_cast<void*>(symbol));
#else
    auto* version = mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol);
#endif

    if (version)
    {
        std::fprintf(
            stderr,
            "MDL SDK version mismatch.\n"
            "  Library: %s\n"
            "  Headers: %s\n",
            version->get_product_version(),
            MI_NEURAYLIB_PRODUCT_VERSION_STRING);
        version->release();
    }
    else
    {
        std::fprintf(stderr, "The MDL SDK library is incompatible with these headers.\n");
    }

    unload();
    return nullptr;
}


bool unload()
{
    if (!g_library)
        return true;

#ifdef MI_PLATFORM_WINDOWS

    if (!FreeLibrary(g_library))
    {
        std::fprintf(stderr, "FreeLibrary() failed for MDL SDK.\n");
        return false;
    }

    g_library = nullptr;
#else

    if (dlclose(g_library) != 0)
    {
        std::fprintf(stderr, "dlclose() failed for MDL SDK: %s\n", dlerror());
        return false;
    }

    g_library = nullptr;
#endif
    return true;
}

} // namespace mdl_loader
