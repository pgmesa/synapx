
#ifndef SYNAPX_CORE_HPP
#define SYNAPX_CORE_HPP

#if defined(_WIN32) || defined(_WIN64)
    #define SYNAPX_API __declspec(dllexport)
#else
    #define SYNAPX_API __attribute__((visibility("default")))
#endif

#endif // SYNAPX_CORE_HPP