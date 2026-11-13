#include "cwru_nn.h"
#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <string.h>

static void error(char *out, size_t cap, const char *reason, int code)
{
    const char *log = CWRUNN_LastLog();
    int used = snprintf(out, cap, "CWRUNN_ERROR,%s,%d,", reason, code);
    if (used < 0 || (size_t)used >= cap) return;
    for (unsigned i = 0U; log[i] != '\0' && i < 159U && cap - (size_t)used > 4U; ++i) {
        int n = snprintf(out + used, cap - (size_t)used, "%02x", (unsigned)(unsigned char)log[i]);
        if (n != 2) return;
        used += n;
    }
    if (cap - (size_t)used >= 3U) (void)snprintf(out + used, cap - (size_t)used, "\r\n");
}

static int nibble(char value)
{
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

void CWRUNN_Command(char *command, char *out, size_t cap)
{
    if (command == NULL || out == NULL || cap == 0U) return;
    if (strcmp(command, "CWRUNN_INFO") == 0) {
        (void)snprintf(out, cap,
            "CWRUNN_INFO,1,14,3,%lu,%u,4064,%08lx,%s,16384,%lu,%d,3d0a0072,-9,3b800000,-128\r\n",
            (unsigned long)HAL_RCC_GetHCLKFreq(), (unsigned)((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U),
            (unsigned long)CWRUNN_ModelFNV(), CWRUNN_MODEL_SHA256,
            (unsigned long)CWRUNN_ArenaUsed(), CWRUNN_Status());
        return;
    }
    if (strcmp(command, "CWRUNN_INIT") == 0) {
        uint32_t cycles; int reused;
        int status = CWRUNN_Init(&cycles, &reused);
        if (status != 0) { error(out, cap, "INIT", status); return; }
        (void)snprintf(out, cap, "CWRUNN_READY,%lu,16384,%lu,%d\r\n",
            (unsigned long)cycles, (unsigned long)CWRUNN_ArenaUsed(), reused);
        return;
    }
    if (strncmp(command, "CWRUNN_RUN,", 11) != 0) { error(out, cap, "UNKNOWN", -1); return; }
    const char *p = command + 11;
    if (strlen(p) != 8U + 1U + 28U || p[8] != ',') { error(out, cap, "FORMAT", -1); return; }
    uint32_t expected = 0U, hash = 2166136261U;
    for (unsigned i = 0U; i < 8U; ++i) {
        int digit = nibble(p[i]);
        if (digit < 0) { error(out, cap, "FORMAT", -2); return; }
        expected = (expected << 4) | (unsigned)digit;
    }
    int8_t input[14], output[3];
    for (unsigned i = 0U; i < 14U; ++i) {
        int high = nibble(p[9U + i*2U]), low = nibble(p[10U + i*2U]);
        if (high < 0 || low < 0) { error(out, cap, "FORMAT", -3); return; }
        uint8_t byte = (uint8_t)((unsigned)high*16U + (unsigned)low);
        memcpy(input + i, &byte, 1U);
        hash = (hash ^ byte) * 16777619U;
    }
    if (hash != expected) { error(out, cap, "CHECKSUM", -1); return; }
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) { error(out, cap, "DWT_DISABLED", -1); return; }
    int predicted; uint32_t invoke_cycles;
    uint32_t start = DWT->CYCCNT;
    int status = CWRUNN_PredictQ(input, output, &predicted, &invoke_cycles);
    uint32_t call_cycles = DWT->CYCCNT - start;
    if (status != 0) { error(out, cap, "INVOKE", status); return; }
    (void)snprintf(out, cap, "CWRUNN_RESULT,%08lx,%lu,%lu,%d,%d,%d,%d,%lu\r\n",
        (unsigned long)hash, (unsigned long)invoke_cycles, (unsigned long)call_cycles,
        predicted, (int)output[0], (int)output[1], (int)output[2], (unsigned long)CWRUNN_ArenaUsed());
}
