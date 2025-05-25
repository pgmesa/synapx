#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <string>

#include <synapx/core.hpp>


namespace synapx {

    class SYNAPX_API Device {
    public:
        // Enumeration for device type
        enum class Type {
            CPU,     // Represents a CPU device
            CUDA     // Represents a CUDA (GPU) device
        };

    private:
        Type _type;  // Stores the type of device (CPU or CUDA)
        int _id;     // For CUDA, this stores the GPU ID (default: 0 for CPU)

    public:
        // Constructor
        explicit Device(Type type = Type::CPU, int id = 0);

        // Accessors
        Type type() const;
        int id() const;

        // Factory methods
        static Device CPU();
        static Device CUDA(int id = 0);

        // Comparison operators
        bool operator==(const Device& other) const;
        bool operator!=(const Device& other) const;

        // String representation of the device
        std::string to_string() const;
    };

} // namespace synapx

#endif // DEVICE_HPP
