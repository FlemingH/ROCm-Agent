#pragma once

#include <vector>
#include <functional>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class BindingRegistry {
public:
    using BindingFunction = std::function<void(pybind11::module&)>;

    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }

    void registerBinding(const std::string& name, BindingFunction func) {
        bindings_.push_back({name, func});
    }

    void applyBindings(pybind11::module& m) {
        for (auto& [name, func] : bindings_) {
            func(m);
        }
    }

private:
    std::vector<std::pair<std::string, BindingFunction>> bindings_;
    BindingRegistry() = default;
};

class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, BindingRegistry::BindingFunction func) {
        BindingRegistry::getInstance().registerBinding(name, func);
    }
};

#define REGISTER_BINDING(name, func) \
    static BindingRegistrar _registrar_##name(#name, [](pybind11::module& m) { func(m); })
