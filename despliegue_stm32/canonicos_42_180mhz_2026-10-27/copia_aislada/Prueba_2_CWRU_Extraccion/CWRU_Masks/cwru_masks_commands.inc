/* Incluir solo al final de cmd_cwru_feat.c con CWRU_MASK_BENCH=1.
 * Comparte ventana, extractor y scratch existentes. No ejecuta DNN/detector. */
#include "cwru_masks.h"
#include <stdarg.h>

static float32_t gmask_output[CWRU_TOTAL_FEATURES];
static uint32_t gmask_base_checksum;
static uint32_t gmask_last_id;
static uint8_t gmask_base_valid;

void CWRUMASK_Invalidate(void) { gmask_base_valid = 0U; }

static void mask_error(char *out, size_t cap, const char *reason)
{
    (void)snprintf(out, cap, "CWRUMASK_ERROR,%s,-1\r\n", reason);
}

static int mask_format(char *out, size_t cap, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int n = vsnprintf(out, cap, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= cap) {
        mask_error(out, cap, "RESPONSE_SIZE");
        return 0;
    }
    return 1;
}

static int mask_checksum_word(const char *p, uint32_t *value)
{
    if (strlen(p) != 8U) return 0;
    uint32_t word = 0U;
    for (unsigned i = 0; i < 8U; ++i) {
        unsigned nibble;
        if (p[i] >= '0' && p[i] <= '9') nibble = (unsigned)(p[i] - '0');
        else if (p[i] >= 'a' && p[i] <= 'f') nibble = (unsigned)(p[i] - 'a' + 10);
        else if (p[i] >= 'A' && p[i] <= 'F') nibble = (unsigned)(p[i] - 'A' + 10);
        else return 0;
        word = (word << 4) | nibble;
    }
    *value = word;
    return 1;
}

static void mask_hex(char *out, const void *pointer, unsigned count)
{
    const unsigned char *p = (const unsigned char *)pointer;
    const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i < count; ++i) {
        out[2U*i] = digits[p[i] >> 4];
        out[2U*i+1U] = digits[p[i] & 15U];
    }
    out[2U*count] = '\0';
}

static int mask_ready(uint32_t checksum, char *out, size_t cap)
{
    if (!g_active || g_received != TOTAL) { mask_error(out, cap, "INCOMPLETE"); return 0; }
    if (checksum != g_checksum) { mask_error(out, cap, "CHECKSUM"); return 0; }
    if (!g_initialized) { mask_error(out, cap, "NOT_INITIALIZED"); return 0; }
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) { mask_error(out, cap, "DWT_DISABLED"); return 0; }
    return 1;
}

static void mask_base(const char *p, char *out, size_t cap)
{
    CWRUMASK_Invalidate();
    uint32_t checksum;
    if (!mask_checksum_word(p, &checksum)) { mask_error(out, cap, "FORMAT"); return; }
    if (!mask_ready(checksum, out, cap)) return;
    /* Limpiar publicaciones fuera de la envolvente temporal. */
    memset(g_features, 0, sizeof(g_features));
    uint32_t start = DWT->CYCCNT;
    int status = CWRU_FeatureExtraction_Extract(&g_extractor, g_window,
        g_features, g_fft_input, g_fft_output);
    uint32_t elapsed = DWT->CYCCNT - start;
    if (status != CWRU_FE_OK) { mask_error(out, cap, "EXTRACT"); return; }
    char hex[113];
    mask_hex(hex, g_features, sizeof(g_features));
    if (mask_format(out, cap, "CWRUMASK_BASE,%08lx,%lu,%s\r\n",
        (unsigned long)checksum, (unsigned long)elapsed, hex)) {
        gmask_base_checksum = checksum;
        gmask_base_valid = 1U;
    }
}

static void mask_run(const char *p, char *out, size_t cap)
{
    if (*p < '0' || *p > '9') { mask_error(out, cap, "FORMAT"); return; }
    const char *q = p;
    while (*q >= '0' && *q <= '9') ++q;
    if (*q != ',') { mask_error(out, cap, "FORMAT"); return; }
    char *end;
    errno = 0;
    unsigned long parsed = strtoul(p, &end, 10);
    if (errno || end != q || parsed > UINT32_MAX) { mask_error(out, cap, "FORMAT"); return; }
    uint32_t checksum;
    if (!mask_checksum_word(q + 1, &checksum)) { mask_error(out, cap, "FORMAT"); return; }
    const CWRUMASK_Spec *spec = CWRUMASK_GetSpec((uint32_t)parsed);
    if (spec == NULL) { mask_error(out, cap, "MASK_ID"); return; }
    if (!mask_ready(checksum, out, cap)) return;
    if (!gmask_base_valid || gmask_base_checksum != checksum) { mask_error(out, cap, "NO_BASE"); return; }
    /* Los slots inactivos son diagnostico: no se cobra su limpieza al kernel. */
    memset(gmask_output, 0, sizeof(gmask_output));
    uint32_t start = DWT->CYCCNT;
    int status = CWRUMASK_Extract((uint32_t)parsed, &g_extractor, g_window,
        gmask_output, g_fft_input, g_fft_output);
    uint32_t elapsed = DWT->CYCCNT - start;
    if (status != CWRU_FE_OK) { mask_error(out, cap, "EXTRACT"); return; }
    uint16_t exact = 0U;
    for (unsigned i = 0; i < CWRU_TOTAL_FEATURES; ++i) {
        uint16_t bit = (uint16_t)(1U << i);
        if ((spec->feature_mask & bit) && !memcmp(&gmask_output[i], &g_features[i], sizeof(float32_t))) exact |= bit;
    }
    char hex[113];
    mask_hex(hex, gmask_output, sizeof(gmask_output));
    uint32_t next = gmask_last_id + 1U;
    if (next == 0U) next = 1U;
    if (mask_format(out, cap, "CWRUMASK_RESULT,%lu,%08lx,%lu,%04x,%lu,%04x,%s\r\n",
        (unsigned long)next, (unsigned long)checksum, parsed,
        (unsigned)spec->feature_mask, (unsigned long)elapsed, (unsigned)exact, hex)) gmask_last_id = next;
}

void CWRUMASK_Command(char *command, char *out, size_t cap)
{
    if (command == NULL || out == NULL || cap == 0U) return;
    if (!strcmp(command, "CWRUMASK_INFO")) {
        mask_format(out, cap, "CWRUMASK_INFO,1,12000,4096,2,14,%lu,%u,%08lx,10,2,1,%08lx,%u,%lu\r\n",
            (unsigned long)HAL_RCC_GetHCLKFreq(), (unsigned)((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U),
            (unsigned long)__get_FPSCR(), (unsigned long)CWRUMASK_CatalogFNV(),
            (unsigned)gmask_base_valid, (unsigned long)gmask_last_id);
        return;
    }
    if (!strncmp(command, "CWRUMASK_INFO,", 14)) { mask_error(out, cap, "FORMAT"); return; }
    if (!strcmp(command, "CWRUMASK_BASE")) { mask_base("", out, cap); return; }
    if (!strncmp(command, "CWRUMASK_BASE,", 14)) { mask_base(command + 14, out, cap); return; }
    if (!strcmp(command, "CWRUMASK_RUN")) { mask_run("", out, cap); return; }
    if (!strncmp(command, "CWRUMASK_RUN,", 13)) { mask_run(command + 13, out, cap); return; }
    mask_error(out, cap, "UNKNOWN");
}
