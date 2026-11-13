/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_extraction.h"
#include "tinyml_inference.h"
#include "kmeans_anomaly.h"

void CmdRawAll_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
{
    /* ====================================================
     * VARIABLES DE ENTRADA
     * ==================================================== */

    float window[
        FE_WINDOW_SIZE
    ][
        FE_AXIS_COUNT
    ];

    int cantidad = 0;


    char *token =
        strtok(
            comando + 7,
            ","
        );


    /* ====================================================
     * PARSING ESTRICTO DE 48 VALORES
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


        /*
         * Validar token numérico completo.
         */
        if (
            fin == token
            ||
            *fin != '\0'
        )
        {
            snprintf(
                respuesta,
                capacidad_respuesta,
                "RAWALL_ERROR,VALOR,%d\r\n",
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
            "RAWALL_ERROR,CANTIDAD,%d\r\n",
            cantidad
        );

        return;
    }


    /*
     * No permitir valores adicionales.
     */
    if (token != NULL)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "RAWALL_ERROR,EXTRA\r\n"
        );

        return;
    }


    /* ====================================================
     * 1. FEATURE EXTRACTION
     *
     * IMPORTANTE:
     *
     * Se ejecuta UNA SOLA VEZ.
     *
     * Las mismas 21 features alimentan:
     *
     * - DNN
     * - K-means
     * ==================================================== */

    float features[
        FE_TOTAL_FEATURES
    ];


    int estado_features =
        FeatureExtraction_Extract(
            window,
            features
        );


    if (estado_features != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "RAWALL_ERROR,FEATURES,%d\r\n",
            estado_features
        );

        return;
    }


    /* ====================================================
     * 2. DNN TINYML
     * ==================================================== */

    int clase = -1;


    float probs[3] =
    {
        0.0f,
        0.0f,
        0.0f
    };


    int estado_pred =
        TinyML_Predict(
            features,
            &clase,
            probs
        );


    if (estado_pred != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "RAWALL_ERROR,PRED,%d\r\n",
            estado_pred
        );

        return;
    }


    /* ====================================================
     * 3. K-MEANS
     * ==================================================== */

    KMeansResult resultado_kmeans;


    int estado_kmeans =
        KMeans_Compute(
            features,
            &resultado_kmeans
        );


    if (estado_kmeans != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "RAWALL_ERROR,KMEANS,%d\r\n",
            estado_kmeans
        );

        return;
    }


    /* ====================================================
     * RESPUESTA COMBINADA
     *
     * RAWALL,
     * clase,
     * p0,p1,p2,
     * cluster,
     * score_d2,
     * anomaly
     * ==================================================== */

    snprintf(
        respuesta,
        capacidad_respuesta,
        "RAWALL,%d,%.6f,%.6f,%.6f,%d,%.9f,%d\r\n",
        clase,
        probs[0],
        probs[1],
        probs[2],
        resultado_kmeans.cluster,
        resultado_kmeans.score_d2,
        resultado_kmeans.anomaly
    );
}

