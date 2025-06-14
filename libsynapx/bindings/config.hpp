#ifndef BINDINGS_CONFIG_HPP
#define BINDINGS_CONFIG_HPP

#include <spdlog/spdlog.h>


class TensorReprConfig {
public:
    static bool detailed_repr;
    
    static void set_detailed_repr(bool value) {
        detailed_repr = value;
    }
    
    static bool is_detailed_repr_enabled() {
        return detailed_repr;
    }
};

enum class LogLevel {
    DEBUG_LEVEL,
    INFO_LEVEL,
    WARNING_LEVEL,
    ERROR_LEVEL,
    NONE_LEVEL
};


class LoggingConfig {
public:
    static LogLevel debug_level;
    
    static LogLevel get_log_level() {
        return debug_level;
    }

    static void set_log_level(LogLevel level) {
        LogLevel prev_level = debug_level;
        debug_level = level;
        
        switch (level) {
            case LogLevel::DEBUG_LEVEL:
                spdlog::set_level(spdlog::level::debug);
                break;
            case LogLevel::INFO_LEVEL:
                spdlog::set_level(spdlog::level::info);
                break;
            case LogLevel::WARNING_LEVEL:
                spdlog::set_level(spdlog::level::warn);
                break;
            case LogLevel::ERROR_LEVEL:
                spdlog::set_level(spdlog::level::err);
                break;
            case LogLevel::NONE_LEVEL:
                spdlog::set_level(spdlog::level::off);
                break;
            default:
                debug_level = prev_level;
                throw std::invalid_argument("Unknown logging level");
        }
    }
};

#endif