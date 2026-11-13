/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_extraction.h"
#include "tinyml_inference.h"

void CmdPred_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
{
    float features[
        FE_TOTAL_FEATURES
    ];


    int cantidad =
        0;


    char *token =
        strtok(
            comando + 5,
            ","
        );


    while (
        token != NULL
        &&
        cantidad
        <
        FE_TOTAL_FEATURES
    )
    {
        features[cantidad] =
            strtof(
                token,
                NULL
            );


        cantidad++;


        token =
            strtok(
                NULL,
                ","
            );
    }


    if (
        cantidad
        !=
        FE_TOTAL_FEATURES
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "PRED_ERROR,%d\r\n",
            cantidad
        );
    }
    else
    {
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


        if (estado_pred == 0)
        {
            snprintf(
                respuesta,
                capacidad_respuesta,
                "PRED,%d,%.6f,%.6f,%.6f\r\n",
                clase,
                probs[0],
                probs[1],
                probs[2]
            );
        }
        else
        {
            snprintf(
                respuesta,
                capacidad_respuesta,
                "PRED_ERROR,%d\r\n",
                estado_pred
            );
        }
    }
}

