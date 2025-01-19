
#include <sstream>

#include <synapx/device.hpp>


namespace synapx {

// Constructor
Device::Device(Type type, int id)
    : _type(type), _id(id) {}

// Accessors
Device::Type Device::type() const {
    return _type;
}

int Device::id() const {
    return _id;
}

// Factory methods
Device Device::CPU() {
    return Device(Type::CPU);
}

Device Device::CUDA(int id) {
    return Device(Type::CUDA, id);
}

// Comparison operators
bool Device::operator==(const Device& other) const {
    return _type == other._type && _id == other._id;
}

bool Device::operator!=(const Device& other) const {
    return !(*this == other);
}

// String representation
std::string Device::to_string() const {
    if (_type == Type::CPU) {
        return "CPU";
    } else {
        return "CUDA:" + std::to_string(_id);
    }
}

} // namespace synapx
