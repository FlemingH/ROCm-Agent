#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
