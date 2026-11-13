/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_extraction.h"
#include "tinyml_inference.h"

void CmdRawPred_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
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
            comando + 8,
            ","
        );


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
         * corresponde a un número.
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
                "RAWPRED_ERROR,VALOR,%d\r\n",
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
     * Deben existir exactamente 48 valores.
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
            "RAWPRED_ERROR,CANTIDAD,%d\r\n",
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
            "RAWPRED_ERROR,EXTRA\r\n"
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
            "RAWPRED_ERROR,FEATURES,%d\r\n",
            estado_features
        );

        return;
    }


    /* ====================================================
     * INFERENCIA
     * ==================================================== */

    int clase =
        -1;


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
            "RAWPRED_ERROR,PRED,%d\r\n",
            estado_pred
        );

        return;
    }


    snprintf(
        respuesta,
        capacidad_respuesta,
        "RAWPRED,%d,%.6f,%.6f,%.6f\r\n",
        clase,
        probs[0],
        probs[1],
        probs[2]
    );
}

