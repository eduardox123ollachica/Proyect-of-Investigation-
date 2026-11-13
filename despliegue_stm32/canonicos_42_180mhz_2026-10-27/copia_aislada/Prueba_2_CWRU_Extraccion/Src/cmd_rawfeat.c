/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_extraction.h"

void CmdRawFeat_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
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


    /*
     * Leer:
     *
     * 16 muestras x 3 ejes = 48 valores.
     */

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


        window[muestra][eje] =
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
            "RAWFEAT_ERROR,%d\r\n",
            cantidad
        );

        return;
    }


    float features[
        FE_TOTAL_FEATURES
    ];


    int estado =
        FeatureExtraction_Extract(
            window,
            features
        );


    if (estado != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "RAWFEAT_ERROR,%d\r\n",
            estado
        );

        return;
    }


    int usados =
        snprintf(
            respuesta,
            capacidad_respuesta,
            "FEATURES"
        );


    for (
        int i = 0;
        i < FE_TOTAL_FEATURES;
        i++
    )
    {
        if (
            usados < 0
            ||
            (size_t)usados
            >=
            capacidad_respuesta
        )
        {
            return;
        }


        int escritos =
            snprintf(
                respuesta + usados,
                capacidad_respuesta - usados,
                ",%.9f",
                features[i]
            );


        if (escritos < 0)
        {
            return;
        }


        usados += escritos;
    }


    if (
        (size_t)usados
        <
        capacidad_respuesta
    )
    {
        snprintf(
            respuesta + usados,
            capacidad_respuesta - usados,
            "\r\n"
        );
    }
}

