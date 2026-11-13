#include "cwru_normality.h"
#include "cwru_normality_params.h"
#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <string.h>

static uint32_t bits(float value) { uint32_t word; memcpy(&word, &value, 4U); return word; }
static int nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static void hex_bytes(char *out, const void *pointer, unsigned size)
{
    const unsigned char *p = (const unsigned char *)pointer;
    const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i < size; ++i) { out[2*i] = digits[p[i] >> 4]; out[2*i+1] = digits[p[i] & 15U]; }
    out[2*size] = '\0';
}
static void error(char *out, size_t cap, const char *reason, int code)
{
    (void)snprintf(out, cap, "CWRUND_ERROR,%s,%d\r\n", reason, code);
}

void CWRUND_Command(char *command, char *out, size_t cap)
{
    if (command == NULL || out == NULL || cap == 0U) return;
    if (strcmp(command, "CWRUND_INFO") == 0) {
        char indices[25]; hex_bytes(indices, cwru_normality_feature_indices, 12U);
        (void)snprintf(out, cap, "CWRUND_INFO,1,14,12,1,%lu,%u,%08lx,%s,%08lx,%08lx,%s\r\n",
            (unsigned long)HAL_RCC_GetHCLKFreq(), (unsigned)((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U),
            (unsigned long)__get_FPSCR(), CWRUND_PARAMS_SHA256, (unsigned long)CWRUND_ParamsFNV(),
            (unsigned long)bits(cwru_normality_threshold_d2), indices);
        return;
    }
    const char *names[] = {"RUN", "DIST", "CHECK"};
    const char *prefixes[] = {"CWRUND_RUN,", "CWRUND_DIST,", "CWRUND_CHECK,"};
    unsigned mode;
    const char *p = NULL;
    for (mode = 0; mode < 3U; ++mode) {
        size_t n = strlen(prefixes[mode]);
        if (strncmp(command, prefixes[mode], n) == 0) { p = command + n; break; }
    }
    if (p == NULL) { error(out, cap, "UNKNOWN", -8); return; }
    unsigned count = mode == CWRUND_RUN ? 14U : mode == CWRUND_DIST ? 12U : 1U;
    if (strlen(p) != 9U + count * 8U || p[8] != ',') { error(out, cap, "FORMAT", -6); return; }
    uint32_t expected = 0U, hash = 2166136261U;
    for (unsigned i = 0; i < 8U; ++i) {
        int digit = nibble(p[i]);
        if (digit < 0) { error(out, cap, "FORMAT", -6); return; }
        expected = (expected << 4) | (unsigned)digit;
    }
    unsigned char bytes[56];
    for (unsigned i = 0; i < count * 4U; ++i) {
        int high = nibble(p[9U + i*2U]), low = nibble(p[10U + i*2U]);
        if (high < 0 || low < 0) { error(out, cap, "FORMAT", -6); return; }
        bytes[i] = (unsigned char)((unsigned)high * 16U + (unsigned)low);
        hash = (hash ^ bytes[i]) * 16777619U;
    }
    if (hash != expected) { error(out, cap, "CHECKSUM", -7); return; }
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) { error(out, cap, "DWT_DISABLED", -5); return; }
    float values[14]; memcpy(values, bytes, count * 4U);
    CWRUND_Result result;
    uint32_t start = DWT->CYCCNT;
    int status = CWRUND_Compute(values, mode, &result);
    uint32_t cycles = DWT->CYCCNT - start;
    if (status != 0) {
        const char *reason = status == -1 ? "NONFINITE_INPUT" : status == -2 ? "NONFINITE_INTERMEDIATE" :
            status == -3 ? "NEGATIVE_SCORE" : status == -4 ? "PARAMETERS" : "FORMAT";
        error(out, cap, reason, status); return;
    }
    char z[97], terms[97];
    hex_bytes(z, result.z, mode == CWRUND_CHECK ? 0U : 48U);
    hex_bytes(terms, result.terms, mode == CWRUND_CHECK ? 0U : 48U);
    (void)snprintf(out, cap, "CWRUND_RESULT,%s,%08lx,%lu,%08lx,%d,0,%08lx,%s,%s\r\n",
        names[mode], (unsigned long)hash, (unsigned long)cycles, (unsigned long)bits(result.score),
        result.rejected, (unsigned long)bits(cwru_normality_threshold_d2), z, terms);
}
