/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stm32f4xx_hal.h"
#include "feature_extraction.h"
#include "tinyml_inference.h"

void CmdBenchRaw_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
{
    float window[
        FE_WINDOW_SIZE
    ][
        FE_AXIS_COUNT
    ];


    int cantidad =
        0;


    char *token =
        strtok(
            comando + 9,
            ","
        );


    /* ====================================================
     * PARSING DE LOS 48 VALORES
     *
     * NO forma parte del benchmark.
     * ==================================================== */

    while (
        token != NULL
        &&
        cantidad
        <
        (
            FE_WINDOW_SIZE
            *
            FE_AXIS_COUNT
        )
    )
    {
        int muestra =
            cantidad
            /
            FE_AXIS_COUNT;


        int eje =
            cantidad
            %
            FE_AXIS_COUNT;


        char *fin =
            NULL;


        float valor =
            strtof(
                token,
                &fin
            );


        if (
            fin == token
            ||
            *fin != '\0'
        )
        {
            snprintf(
                respuesta,
                capacidad_respuesta,
                "BENCHRAW_ERROR,VALOR,%d\r\n",
                cantidad
            );

            return;
        }


        window[muestra][eje] =
            valor;


        cantidad++;


        token =
            strtok(
                NULL,
                ","
            );
    }


    /*
     * Exactamente 48 valores.
     */

    if (
        cantidad
        !=
        (
            FE_WINDOW_SIZE
            *
            FE_AXIS_COUNT
        )
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHRAW_ERROR,CANTIDAD,%d\r\n",
            cantidad
        );

        return;
    }


    /*
     * No permitir datos extra.
     */

    if (token != NULL)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHRAW_ERROR,EXTRA\r\n"
        );

        return;
    }


    /* ====================================================
     * VERIFICAR DWT
     * ==================================================== */

    if (
        (
            DWT->CTRL
            &
            DWT_CTRL_CYCCNTENA_Msk
        )
        ==
        0
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHRAW_ERROR,DWT\r\n"
        );

        return;
    }


    /* ====================================================
     * VARIABLES PIPELINE
     * ==================================================== */

    float features[
        FE_TOTAL_FEATURES
    ];


    int clase =
        -1;


    float probs[3] =
    {
        0.0f,
        0.0f,
        0.0f
    };


    /* ====================================================
     * CONTADORES DWT
     * ==================================================== */

    uint32_t inicio_total;

    uint32_t inicio_features;
    uint32_t fin_features;

    uint32_t inicio_inferencia;
    uint32_t fin_inferencia;

    uint32_t fin_total;


    /* ====================================================
     * INICIO DEL PIPELINE MEDIDO
     * ==================================================== */

    inicio_total =
        DWT->CYCCNT;


    /* ====================================================
     * 1. FEATURE EXTRACTION
     * ==================================================== */

    inicio_features =
        DWT->CYCCNT;


    int estado_features =
        FeatureExtraction_Extract(
            window,
            features
        );


    fin_features =
        DWT->CYCCNT;


    if (estado_features != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHRAW_ERROR,FEATURES,%d\r\n",
            estado_features
        );

        return;
    }


    /* ====================================================
     * 2. INFERENCIA TINYML
     * ==================================================== */

    inicio_inferencia =
        DWT->CYCCNT;


    int estado_pred =
        TinyML_Predict(
            features,
            &clase,
            probs
        );


    fin_inferencia =
        DWT->CYCCNT;


    /*
     * El pipeline termina cuando finaliza
     * la inferencia.
     */

    fin_total =
        fin_inferencia;


    if (estado_pred != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHRAW_ERROR,PRED,%d\r\n",
            estado_pred
        );

        return;
    }


    /* ====================================================
     * CALCULAR CICLOS
     * ==================================================== */

    uint32_t ciclos_features =
        fin_features
        -
        inicio_features;


    uint32_t ciclos_inferencia =
        fin_inferencia
        -
        inicio_inferencia;


    uint32_t ciclos_total =
        fin_total
        -
        inicio_total;


    /* ====================================================
     * RESPUESTA
     *
     * Python convierte:
     *
     * microsegundos =
     * ciclos x 1e6 / HCLK
     * ==================================================== */

    snprintf(
        respuesta,
        capacidad_respuesta,
        "BENCHRAW,%d,%lu,%lu,%lu\r\n",
        clase,
        (unsigned long)ciclos_features,
        (unsigned long)ciclos_inferencia,
        (unsigned long)ciclos_total
    );
}

