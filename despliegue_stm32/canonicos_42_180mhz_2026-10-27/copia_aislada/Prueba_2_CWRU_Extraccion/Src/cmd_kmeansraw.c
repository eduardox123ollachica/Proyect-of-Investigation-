/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_extraction.h"
#include "kmeans_anomaly.h"

void CmdKMeansRaw_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
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
            comando + 10,
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


        char *fin =
            NULL;


        float valor =
            strtof(
                token,
                &fin
            );


        /*
         * Comprobar que el token completo
         * sea un número válido.
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
                "KMEANSRAW_ERROR,VALOR,%d\r\n",
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
     * EXACTAMENTE 48 VALORES
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
            "KMEANSRAW_ERROR,CANTIDAD,%d\r\n",
            cantidad
        );

        return;
    }


    /*
     * Detectar valores adicionales.
     */

    if (token != NULL)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "KMEANSRAW_ERROR,EXTRA\r\n"
        );

        return;
    }


    /* ====================================================
     * FEATURE EXTRACTION
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
            "KMEANSRAW_ERROR,FEATURES,%d\r\n",
            estado_features
        );

        return;
    }


    /* ====================================================
     * K-MEANS
     * ==================================================== */

    KMeansResult resultado;


    int estado_kmeans =
        KMeans_Compute(
            features,
            &resultado
        );


    if (estado_kmeans != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "KMEANSRAW_ERROR,KMEANS,%d\r\n",
            estado_kmeans
        );

        return;
    }


    /* ====================================================
     * RESPUESTA
     * ==================================================== */

    snprintf(
        respuesta,
        capacidad_respuesta,
        "KMEANSRAW,%d,%.9f,%d\r\n",
        resultado.cluster,
        resultado.score_d2,
        resultado.anomaly
    );
}

