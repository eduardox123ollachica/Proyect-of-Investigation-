/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stm32f4xx_hal.h"
#include "feature_extraction.h"
#include "tinyml_inference.h"
#include "kmeans_anomaly.h"

void CmdBenchAll_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
{
    float window[
        FE_WINDOW_SIZE
    ][
        FE_AXIS_COUNT
    ];

    int cantidad = 0;

    char *token =
        strtok(
            comando + 9,
            ","
        );


    /* ====================================================
     * PARSING DE 48 VALORES
     *
     * FUERA DEL BENCHMARK
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

        char *fin = NULL;

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
                "BENCHALL_ERROR,VALOR,%d\r\n",
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


    /* ====================================================
     * VALIDAR CANTIDAD
     * ==================================================== */

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
            "BENCHALL_ERROR,CANTIDAD,%d\r\n",
            cantidad
        );

        return;
    }


    if (token != NULL)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHALL_ERROR,EXTRA\r\n"
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
            "BENCHALL_ERROR,DWT\r\n"
        );

        return;
    }


    /* ====================================================
     * VARIABLES
     * ==================================================== */

    float features[
        FE_TOTAL_FEATURES
    ];

    int clase = -1;

    float probs[3] =
    {
        0.0f,
        0.0f,
        0.0f
    };

    KMeansResult resultado_kmeans;


    /* ====================================================
     * CONTADORES DWT
     * ==================================================== */

    uint32_t inicio_total;
    uint32_t fin_total;

    uint32_t inicio_features;
    uint32_t fin_features;

    uint32_t inicio_dnn;
    uint32_t fin_dnn;

    uint32_t inicio_kmeans;
    uint32_t fin_kmeans;


    /* ====================================================
     * INICIO PIPELINE
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
            "BENCHALL_ERROR,FEATURES,%d\r\n",
            estado_features
        );

        return;
    }


    /* ====================================================
     * 2. DNN
     * ==================================================== */

    inicio_dnn =
        DWT->CYCCNT;

    int estado_pred =
        TinyML_Predict(
            features,
            &clase,
            probs
        );

    fin_dnn =
        DWT->CYCCNT;


    if (estado_pred != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHALL_ERROR,PRED,%d\r\n",
            estado_pred
        );

        return;
    }


    /* ====================================================
     * 3. K-MEANS
     * ==================================================== */

    inicio_kmeans =
        DWT->CYCCNT;

    int estado_kmeans =
        KMeans_Compute(
            features,
            &resultado_kmeans
        );

    fin_kmeans =
        DWT->CYCCNT;


    if (estado_kmeans != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHALL_ERROR,KMEANS,%d\r\n",
            estado_kmeans
        );

        return;
    }


    /* ====================================================
     * FIN PIPELINE
     * ==================================================== */

    fin_total =
        DWT->CYCCNT;


    /* ====================================================
     * CICLOS
     * ==================================================== */

    uint32_t ciclos_features =
        fin_features
        -
        inicio_features;


    uint32_t ciclos_dnn =
        fin_dnn
        -
        inicio_dnn;


    uint32_t ciclos_kmeans =
        fin_kmeans
        -
        inicio_kmeans;


    uint32_t ciclos_total =
        fin_total
        -
        inicio_total;


    /* ====================================================
     * RESPUESTA
     * ==================================================== */

    snprintf(
        respuesta,
        capacidad_respuesta,
        "BENCHALL,%d,%d,%lu,%lu,%lu,%lu\r\n",
        clase,
        resultado_kmeans.anomaly,
        (unsigned long)ciclos_features,
        (unsigned long)ciclos_dnn,
        (unsigned long)ciclos_kmeans,
        (unsigned long)ciclos_total
    );
}

