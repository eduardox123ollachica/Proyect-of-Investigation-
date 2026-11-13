#include "cwru_total_nn.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_log.h"
#include <cstdio>
#include <cstring>

namespace { char last_log[160]; }
extern "C" void DebugLog(const char *format, va_list args)
{
    if (format != nullptr && std::strcmp(format, "\r\n") != 0 && std::strcmp(format, "\n") != 0)
        (void)std::vsnprintf(last_log, sizeof(last_log), format, args);
}
extern "C" int DebugVsnprintf(char *buffer, size_t capacity, const char *format, va_list args)
{
    return std::vsnprintf(buffer, capacity, format, args);
}
extern "C" const char *CWRUTNN_LastLog(void) { return last_log; }

/* Las cabeceras usan estas interfaces, ausentes del archivo .a disponible.
 * Mantener los diagnósticos en memoria, nunca transmitir durante Invoke. */
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
void VMicroPrintf(const char *format, va_list args) { DebugLog(format, args); }
void MicroPrintf(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    VMicroPrintf(format, args);
    va_end(args);
}
int MicroVsnprintf(char *buffer, size_t capacity, const char *format, va_list args)
{
    return DebugVsnprintf(buffer, capacity, format, args);
}
int MicroSnprintf(char *buffer, size_t capacity, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int result = MicroVsnprintf(buffer, capacity, format, args);
    va_end(args);
    return result;
}
#endif
