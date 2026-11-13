#include "uart_commands.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uart_command_handlers.h"

void ProcesarComandoUART(
    char *comando,
    int estado_tinyml,
    char *respuesta,
    size_t capacidad_respuesta
)
{
    if (strcmp(comando, "MODELO") == 0)
    {
        CmdModelo_Execute(respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "KMEANSRAW,", 10) == 0)
    {
        CmdKMeansRaw_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "RAWALL,", 7) == 0)
    {
        CmdRawAll_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "BENCHALL,", 9) == 0)
    {
        CmdBenchAll_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "STACKBENCH,", 11) == 0)
    {
        CmdStackBench_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strcmp(comando, "TINYML") == 0)
    {
        CmdTinyML_Execute(estado_tinyml, respuesta, capacidad_respuesta);
    }
    else if (strcmp(comando, "INFO") == 0)
    {
        CmdInfo_Execute(respuesta, capacidad_respuesta);
    }
    else if (strcmp(comando, "QUANT") == 0)
    {
        CmdQuant_Execute(respuesta, capacidad_respuesta);
    }
    else if (strcmp(comando, "BENCHINFO") == 0)
    {
        CmdBenchInfo_Execute(respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "RAWFEAT,", 8) == 0)
    {
        CmdRawFeat_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "RAWPRED,", 8) == 0)
    {
        CmdRawPred_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "BENCHRAW,", 9) == 0)
    {
        CmdBenchRaw_Execute(comando, respuesta, capacidad_respuesta);
    }
    else if (strncmp(comando, "PRED,", 5) == 0)
    {
        CmdPred_Execute(comando, respuesta, capacidad_respuesta);
    }
    else
    {
        int valor = atoi(comando);

        snprintf(
            respuesta,
            capacidad_respuesta,
            "RECIBIDO,%d\r\n",
            valor
        );
    }
}
