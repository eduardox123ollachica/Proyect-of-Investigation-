/*
 * Comando UART aislado para validar la extracción CWRU PC--STM32.
 *
 * Protocolo:
 *   CWRUFEAT_BEGIN
 *   CWRUFEAT_DATA,<offset>,<valor0>,...,<valorN>
 *   CWRUFEAT_RUN
 *
 * Los datos se transmiten aplanados por muestra:
 *   DE[0], FE[0], DE[1], FE[1], ..., DE[4095], FE[4095].
 *
 * Este archivo está preparado para copiarse a Src/ de "Prueba 2".
 */

#include "uart_command_handlers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwru_feature_extraction.h"
#include "stm32f4xx_hal.h"

#define CWRU_FE_TOTAL_INPUT_VALUES \
    (CWRU_WINDOW_SAMPLES * CWRU_NUM_CHANNELS)

/*
 * Memoria estática deliberada:
 *   ventana dual:       32768 bytes
 *   scratch RFFT:       32768 bytes
 *   features/contexto:  menos de 1 KiB
 *
 * No mover estos arreglos al interior de la función: desbordarían el stack.
 */
static float32_t g_cwru_window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS];
static float32_t g_cwru_fft_input[CWRU_FE_SCRATCH_FLOATS];
static float32_t g_cwru_fft_output[CWRU_FE_SCRATCH_FLOATS];
static float32_t g_cwru_features[CWRU_TOTAL_FEATURES];
static CWRU_FeatureExtractor g_cwru_extractor;

static uint32_t g_received_values = 0U;
static uint8_t g_extractor_initialized = 0U;

static void respond_error(
    char *response,
    size_t response_capacity,
    const char *reason,
    long detail
)
{
    (void)snprintf(
        response,
        response_capacity,
        "CWRUFEAT_ERROR,%s,%ld\r\n",
        reason,
        detail
    );
}

static int ensure_extractor_initialized(void)
{
    int status;

    if (g_extractor_initialized != 0U)
    {
        return CWRU_FE_OK;
    }

    status = CWRU_FeatureExtraction_Init(&g_cwru_extractor);
    if (status == CWRU_FE_OK)
    {
        g_extractor_initialized = 1U;
    }

    return status;
}

static void execute_begin(char *response, size_t response_capacity)
{
    const int status = ensure_extractor_initialized();

    if (status != CWRU_FE_OK)
    {
        respond_error(response, response_capacity, "INIT", status);
        return;
    }

    g_received_values = 0U;

    (void)snprintf(
        response,
        response_capacity,
        "CWRUFEAT_READY,%lu\r\n",
        (unsigned long)CWRU_FE_TOTAL_INPUT_VALUES
    );
}

static void execute_data(
    char *command,
    char *response,
    size_t response_capacity
)
{
    char *token;
    char *end_pointer;
    unsigned long offset;
    uint32_t values_in_frame = 0U;

    /* Saltar "CWRUFEAT_DATA,". */
    token = strtok(command + 14, ",");
    if (token == NULL)
    {
        respond_error(response, response_capacity, "NO_OFFSET", -1L);
        return;
    }

    offset = strtoul(token, &end_pointer, 10);
    if ((*token == '\0') || (*end_pointer != '\0'))
    {
        respond_error(response, response_capacity, "BAD_OFFSET", -1L);
        return;
    }

    if ((uint32_t)offset != g_received_values)
    {
        respond_error(
            response,
            response_capacity,
            "OFFSET",
            (long)g_received_values
        );
        return;
    }

    while ((token = strtok(NULL, ",")) != NULL)
    {
        float value;
        uint32_t sample;
        uint32_t channel;

        if (g_received_values >= CWRU_FE_TOTAL_INPUT_VALUES)
        {
            respond_error(response, response_capacity, "TOO_MANY", -1L);
            return;
        }

        value = strtof(token, &end_pointer);
        if ((*token == '\0') || (*end_pointer != '\0') || !isfinite(value))
        {
            respond_error(
                response,
                response_capacity,
                "BAD_VALUE",
                (long)g_received_values
            );
            return;
        }

        sample = g_received_values / CWRU_NUM_CHANNELS;
        channel = g_received_values % CWRU_NUM_CHANNELS;
        g_cwru_window[sample][channel] = value;

        ++g_received_values;
        ++values_in_frame;
    }

    if (values_in_frame == 0U)
    {
        respond_error(response, response_capacity, "NO_DATA", -1L);
        return;
    }

    (void)snprintf(
        response,
        response_capacity,
        "CWRUFEAT_ACK,%lu\r\n",
        (unsigned long)g_received_values
    );
}

static void execute_run(char *response, size_t response_capacity)
{
    uint32_t start_cycles;
    uint32_t elapsed_cycles;
    int status;
    int used;
    uint32_t index;

    if (g_received_values != CWRU_FE_TOTAL_INPUT_VALUES)
    {
        respond_error(
            response,
            response_capacity,
            "INCOMPLETE",
            (long)g_received_values
        );
        return;
    }

    status = ensure_extractor_initialized();
    if (status != CWRU_FE_OK)
    {
        respond_error(response, response_capacity, "INIT", status);
        return;
    }

    start_cycles = DWT->CYCCNT;
    status = CWRU_FeatureExtraction_Extract(
        &g_cwru_extractor,
        g_cwru_window,
        g_cwru_features,
        g_cwru_fft_input,
        g_cwru_fft_output
    );
    elapsed_cycles = DWT->CYCCNT - start_cycles;

    if (status != CWRU_FE_OK)
    {
        respond_error(response, response_capacity, "EXTRACT", status);
        return;
    }

    used = snprintf(
        response,
        response_capacity,
        "CWRUFEAT_RESULT,%lu",
        (unsigned long)elapsed_cycles
    );

    if ((used < 0) || ((size_t)used >= response_capacity))
    {
        return;
    }

    for (index = 0U; index < CWRU_TOTAL_FEATURES; ++index)
    {
        const int written = snprintf(
            response + used,
            response_capacity - (size_t)used,
            ",%.9g",
            (double)g_cwru_features[index]
        );

        if ((written < 0) ||
            ((size_t)written >= (response_capacity - (size_t)used)))
        {
            return;
        }

        used += written;
    }

    if ((size_t)used < response_capacity)
    {
        (void)snprintf(
            response + used,
            response_capacity - (size_t)used,
            "\r\n"
        );
    }
}

void CmdCwruFeat_Execute(
    char *command,
    char *response,
    size_t response_capacity
)
{
    if ((command == NULL) || (response == NULL) || (response_capacity == 0U))
    {
        return;
    }

    if (strcmp(command, "CWRUFEAT_BEGIN") == 0)
    {
        execute_begin(response, response_capacity);
    }
    else if (strncmp(command, "CWRUFEAT_DATA,", 14) == 0)
    {
        execute_data(command, response, response_capacity);
    }
    else if (strcmp(command, "CWRUFEAT_RUN") == 0)
    {
        execute_run(response, response_capacity);
    }
    else
    {
        respond_error(response, response_capacity, "UNKNOWN", -1L);
    }
}
