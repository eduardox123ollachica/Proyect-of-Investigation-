#include "uart_commands.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uart_command_handlers.h"
#include "cwru_pipeline_config.h"

#ifndef CWRU_PIPELINE_TEST
#define CWRU_PIPELINE_TEST 0
#endif
#if CWRU_PIPELINE_TEST
#include "cwru_pipeline.h"
#endif

#ifndef CWRU_STACK_TEST
#define CWRU_STACK_TEST 0
#endif
#if CWRU_STACK_TEST
#include "cwru_stack.h"
#endif

#ifndef CWRU_COST_PROFILE
#define CWRU_COST_PROFILE 0
#endif
#if CWRU_COST_PROFILE
#include "cwru_cost.h"
#endif


#ifndef CWRU_TOTALV2_TEST
#define CWRU_TOTALV2_TEST 0
#endif
#if CWRU_TOTALV2_TEST
#include "cwru_total.h"
#endif

#ifndef CWRU_MASK_BENCH
#define CWRU_MASK_BENCH 0
#endif
#if CWRU_MASK_BENCH
#include "cwru_masks.h"
#endif

#ifndef CWRU_NORMALITY_TEST
#define CWRU_NORMALITY_TEST 0
#endif
#if CWRU_NORMALITY_TEST
#include "cwru_normality.h"
#endif

#ifndef CWRU_DNN_TEST
#define CWRU_DNN_TEST 0
#endif
#if CWRU_DNN_TEST
#include "cwru_nn.h"
#endif

#ifndef CWRU_INPUT_TRANSFORM
#define CWRU_INPUT_TRANSFORM 0
#endif
#if CWRU_INPUT_TRANSFORM
#include "cwru_input_transform.h"
#endif

void ProcesarComandoUART(
    char *comando,
    int estado_tinyml,
    char *respuesta,
    size_t capacidad_respuesta
)
{
#if CWRU_TOTALV2_TEST
    if (strncmp(comando, "CWRUTOTAL_", 10) == 0)
    {
        CWRUTOTAL_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_MASK_BENCH
    if (strncmp(comando, "CWRUMASK_", 9) == 0)
    {
        CWRUMASK_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_COST_PROFILE
    if (strncmp(comando, "CWRUCOST_", 9) == 0)
    {
        CWRUCOST_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_STACK_TEST
    if (strncmp(comando, "CWRUSTK_", 8) == 0)
    {
        CWRUSTK_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_PIPELINE_TEST
    if (strncmp(comando, "CWRUPIPE_", 9) == 0)
    {
        CWRUPIPE_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_NORMALITY_TEST
    if (strncmp(comando, "CWRUND_", 7) == 0)
    {
        CWRUND_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_DNN_TEST
    if (strncmp(comando, "CWRUNN_", 7) == 0)
    {
        CWRUNN_Command(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
#if CWRU_INPUT_TRANSFORM
    if (strncmp(comando, "CWRUIN_", 7) == 0)
    {
        CWRU_InputCommand(comando, respuesta, capacidad_respuesta);
        return;
    }
#endif
    if (strncmp(comando, "CWRUFEAT_", 9) == 0)
    {
        CmdCwruFeat_Execute(comando, respuesta, capacidad_respuesta);
        return;
    }
#if CWRU_EXTRACTION_ONLY
    (void)estado_tinyml;
    snprintf(respuesta, capacidad_respuesta,
             "CWRUFEAT_ERROR,EXTRACTION_ONLY,-1\r\n");
#else
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
#endif
}
