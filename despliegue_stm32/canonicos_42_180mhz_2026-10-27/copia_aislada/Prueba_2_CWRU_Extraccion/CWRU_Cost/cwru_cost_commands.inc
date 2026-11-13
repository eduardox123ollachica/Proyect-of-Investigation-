/* Incluir despues de cwru_pipeline_commands.inc, solo con COST_PROFILE=1.
 * Lee los 14 contadores existentes: no repite ni modifica la extraccion. */
#include "cwru_cost.h"

_Static_assert(CWRU_PROFILE_BLOCKS == 7, "Orden de bloques incompatible.");
_Static_assert(CWRU_NUM_CHANNELS == 2, "Se esperan DE y FE.");

static void cost_error(char *out, size_t cap, const char *reason)
{
    (void)snprintf(out, cap, "CWRUCOST_ERROR,%s,-1\r\n", reason);
}

static int cost_format(char *out, size_t cap, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int n = vsnprintf(out, cap, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= cap) {
        cost_error(out, cap, "RESPONSE_SIZE");
        return 0;
    }
    return 1;
}

void CWRUCOST_Command(char *command, char *out, size_t cap)
{
    if (command == NULL || out == NULL || cap == 0U) return;
    if (!strcmp(command, "CWRUCOST_INFO")) {
        if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) {
            cost_error(out, cap, "DWT_DISABLED");
            return;
        }
        const uint32_t paired_min = CWRU_FeatureExtraction_ProfileReadOverhead();
        cost_format(out, cap, "CWRUCOST_INFO,1,7,2,%lu,1,2,1,%lu,0,64\r\n",
            (unsigned long)HAL_RCC_GetHCLKFreq(), (unsigned long)paired_min);
        return;
    }
    if (!strncmp(command, "CWRUCOST_INFO,", 14) ||
        !strcmp(command, "CWRUCOST_GET")) {
        cost_error(out, cap, "FORMAT");
        return;
    }
    if (strncmp(command, "CWRUCOST_GET,", 13)) {
        cost_error(out, cap, "UNKNOWN");
        return;
    }
    uint32_t id;
    if (!pipe_record_id(command + 13, &id)) {
        cost_error(out, cap, "FORMAT");
        return;
    }
    if (!g_pipe_valid) {
        cost_error(out, cap, "NO_RESULT");
        return;
    }
    if (id != g_pipe_id) {
        cost_error(out, cap, "STALE_ID");
        return;
    }
    const CWRU_ExtractionProfile *profile = CWRU_FeatureExtraction_GetProfile();
    const uint32_t *de = profile->cycles[0];
    const uint32_t *fe = profile->cycles[1];
    cost_format(out, cap,
        "CWRUCOST_GET,%lu,%08lx,7,2,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
        (unsigned long)id, (unsigned long)g_pipe_checksum,
        (unsigned long)de[0], (unsigned long)de[1], (unsigned long)de[2],
        (unsigned long)de[3], (unsigned long)de[4], (unsigned long)de[5],
        (unsigned long)de[6], (unsigned long)fe[0], (unsigned long)fe[1],
        (unsigned long)fe[2], (unsigned long)fe[3], (unsigned long)fe[4],
        (unsigned long)fe[5], (unsigned long)fe[6]);
}
