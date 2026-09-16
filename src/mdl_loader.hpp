#pragma once

#include <mi/mdl_sdk.h>

namespace mdl_loader
{

// Loads libmdl_sdk / mdl_sdk.dll and returns the MDL SDK's
// main INeuray interface.
// The returned interface owns a reference and should be held
// in mi::base::Handle.
mi::neuraylib::INeuray* load(const char* filename = nullptr);

// Unloads the MDL SDK shared library.
// All MDL SDK interfaces must have been released before
// calling this.
bool unload();

} // namespace mdl_loader
