#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(hip_extension, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
