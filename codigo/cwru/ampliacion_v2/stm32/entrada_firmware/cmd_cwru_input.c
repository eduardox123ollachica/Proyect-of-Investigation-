#include "cwru_input_transform.h"
#include "cwru_dnn_scaler.h"
#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <string.h>

static uint32_t float_bits(float value)
{
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

/* Ocho dígitos EXACTOS, con coma o fin; sin signo, espacios ni prefijo 0x. */
static int hex_word(const char **cursor, uint32_t *result, int last)
{
    const char *p = *cursor;
    uint32_t value = 0U;
    for (unsigned i = 0U; i < 8U; ++i) {
        unsigned digit;
        char c = p[i];
        if (c >= '0' && c <= '9') digit = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') digit = 10U + (unsigned)(c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10U + (unsigned)(c - 'A');
        else return 0;
        value = (value << 4) | digit;
    }
    if (p[8] != (last ? '\0' : ',')) return 0;
    *result = value;
    *cursor = p + 8 + (last ? 0 : 1);
    return 1;
}

static void error(char *out, size_t cap, const char *reason, int detail)
{
    (void)snprintf(out, cap, "CWRUIN_ERROR,%s,%d\r\n", reason, detail);
}

void CWRU_InputCommand(char *command, char *response, size_t capacity)
{
    if (command == NULL || response == NULL || capacity == 0U) return;
    if (strcmp(command, "CWRUIN_INFO") == 0) {
        (void)snprintf(response, capacity, "CWRUIN_INFO,1,14,%lu,%u,%08lx,%d,%08lx,%s\r\n",
            (unsigned long)HAL_RCC_GetHCLKFreq(),
            (unsigned)((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U),
            (unsigned long)float_bits(cwru_dnn_input_quant_scale),
            cwru_dnn_input_quant_zero_point, (unsigned long)__get_FPSCR(),
            CWRU_INPUT_PARAMS_SHA256);
        return;
    }
    unsigned mode;
    const char *p;
    if (strncmp(command, "CWRUIN_RUN,", 11) == 0) { mode = 0U; p = command + 11; }
    else if (strncmp(command, "CWRUIN_QUANT,", 13) == 0) { mode = 1U; p = command + 13; }
    else if (strncmp(command, "CWRUIN_ROUND,", 13) == 0) { mode = 2U; p = command + 13; }
    else { error(response, capacity, "UNKNOWN", -1); return; }
    /* 8 FNV + ',' + 14 palabras de 8 hex, separadas por 13 comas. */
    if (strlen(p) != 8U + 1U + 14U * 8U + 13U) {
        error(response, capacity, "FORMAT", -1); return;
    }
    uint32_t expected, checksum = 2166136261U;
    float input[CWRU_INPUT_FEATURES];
    if (!hex_word(&p, &expected, 0)) { error(response, capacity, "FORMAT", -2); return; }
    for (unsigned j = 0U; j < CWRU_INPUT_FEATURES; ++j) {
        uint32_t word;
        if (!hex_word(&p, &word, j + 1U == CWRU_INPUT_FEATURES)) {
            error(response, capacity, "FORMAT", (int)j); return;
        }
        /* Hash de bytes en orden little-endian, independiente de la notación textual. */
        for (unsigned byte = 0U; byte < 4U; ++byte)
            checksum = (checksum ^ ((word >> (8U * byte)) & 255U)) * 16777619U;
        memcpy(&input[j], &word, sizeof(word));
    }
    if (checksum != expected) { error(response, capacity, "CHECKSUM", -1); return; }
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) {
        error(response, capacity, "DWT_DISABLED", -1); return;
    }
    CWRU_InputResult result;
    uint32_t start = DWT->CYCCNT;
    int status = mode == 0U ? CWRU_InputTransform(input, &result) :
                 mode == 1U ? CWRU_InputQuantize(input, &result) : CWRU_InputRound(input, &result);
    uint32_t elapsed = DWT->CYCCNT - start;
    if (status != CWRU_INPUT_OK) {
        error(response, capacity, status == CWRU_INPUT_NONFINITE ? "NONFINITE" : "INTERMEDIATE", status);
        return;
    }
    int used = snprintf(response, capacity, "CWRUIN_RESULT,%u,%08lx,%lu,%u,%u",
                        mode, (unsigned long)checksum, (unsigned long)elapsed,
                        (unsigned)result.saturated_low, (unsigned)result.saturated_high);
    if (used < 0 || (size_t)used >= capacity) { error(response, capacity, "SIZE", -1); return; }
    for (unsigned block = 0U; block < 3U; ++block) {
        for (unsigned j = 0U; j < CWRU_INPUT_FEATURES; ++j) {
            int written = block < 2U ?
                snprintf(response + used, capacity - (size_t)used, ",%08lx",
                    (unsigned long)float_bits(block == 0U ? result.standardized[j] : result.affine[j])) :
                snprintf(response + used, capacity - (size_t)used, ",%d", (int)result.quantized[j]);
            if (written < 0 || (size_t)written >= capacity - (size_t)used) {
                error(response, capacity, "SIZE", -2); return;
            }
            used += written;
        }
    }
    if (capacity - (size_t)used < 3U) { error(response, capacity, "SIZE", -3); return; }
    (void)snprintf(response + used, capacity - (size_t)used, "\r\n");
}
