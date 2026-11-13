/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_extraction.h"
#include "tinyml_inference.h"
#include "kmeans_anomaly.h"
#include "stack_monitor.h"

/* ============================================================
 * STACK BENCHMARK
 *
 * Contextos y wrappers utilizados únicamente para medir
 * el consumo dinámico de stack de cada etapa.
 *
 * IMPORTANTE:
 *
 * El parsing UART NO se mide.
 *
 * Las estructuras window/features/probs/etc. son creadas
 * por ProcesarComandoUART() antes de iniciar StackMonitor.
 *
 * Por tanto medimos principalmente el stack adicional
 * generado por:
 *
 *   FeatureExtraction_Extract()
 *   TinyML_Predict()
 *   KMeans_Compute()
 *
 * y por sus funciones internas.
 * ============================================================ */


/* ============================================================
 * FEATURES
 * ============================================================ */

typedef struct
{
    float
        (*window)
        [FE_AXIS_COUNT];

    float *features;

} StackBenchFeaturesContext;


static int StackBench_RunFeatures(
    void *context
)
{
    StackBenchFeaturesContext *ctx =
        (StackBenchFeaturesContext *)context;


    return FeatureExtraction_Extract(
        ctx->window,
        ctx->features
    );
}


/* ============================================================
 * DNN
 * ============================================================ */

typedef struct
{
    const float *features;

    int *clase;

    float *probs;

} StackBenchDNNContext;


static int StackBench_RunDNN(
    void *context
)
{
    StackBenchDNNContext *ctx =
        (StackBenchDNNContext *)context;


    return TinyML_Predict(
        ctx->features,
        ctx->clase,
        ctx->probs
    );
}


/* ============================================================
 * K-MEANS
 * ============================================================ */

typedef struct
{
    const float *features;

    KMeansResult *resultado;

} StackBenchKMeansContext;


static int StackBench_RunKMeans(
    void *context
)
{
    StackBenchKMeansContext *ctx =
        (StackBenchKMeansContext *)context;


    return KMeans_Compute(
        ctx->features,
        ctx->resultado
    );
}


/* ============================================================
 * PIPELINE COMPLETO
 * ============================================================ */

typedef struct
{
    float
        (*window)
        [FE_AXIS_COUNT];


    float *features;


    int *clase;


    float *probs;


    KMeansResult *resultado_kmeans;

} StackBenchAllContext;


static int StackBench_RunAll(
    void *context
)
{
    StackBenchAllContext *ctx =
        (StackBenchAllContext *)context;


    /* ========================================================
     * 1. FEATURES
     * ======================================================== */

    int estado_features =
        FeatureExtraction_Extract(
            ctx->window,
            ctx->features
        );


    if (estado_features != 0)
    {
        return
            1000
            +
            estado_features;
    }


    /* ========================================================
     * 2. DNN
     * ======================================================== */

    int estado_dnn =
        TinyML_Predict(
            ctx->features,
            ctx->clase,
            ctx->probs
        );


    if (estado_dnn != 0)
    {
        return
            2000
            +
            estado_dnn;
    }


    /* ========================================================
     * 3. K-MEANS
     * ======================================================== */

    int estado_kmeans =
        KMeans_Compute(
            ctx->features,
            ctx->resultado_kmeans
        );


    if (estado_kmeans != 0)
    {
        return
            3000
            +
            estado_kmeans;
    }


    return 0;
}


/* ============================================================
 * BASELINE
 *
 * Sirve para determinar cuánto stack introduce únicamente
 * la llamada mediante StackMonitorFunction.
 *
 * No realiza procesamiento.
 * ============================================================ */

static int StackBench_RunBaseline(
    void *context
)
{
    (void)context;

    return 0;
}
void CmdStackBench_Execute(char *comando, char *respuesta, size_t capacidad_respuesta)
{
    /* ====================================================
     * EXTRAER MODO
     * ==================================================== */

    char *payload =
        comando
        +
        11;


    char *coma =
        strchr(
            payload,
            ','
        );


    if (coma == NULL)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "STACKBENCH_ERROR,MODO\r\n"
        );

        return;
    }


    /*
     * Separar:
     *
     * FEATURES
     * DNN
     * KMEANS
     * ALL
     */
    *coma =
        '\0';


    char *modo =
        payload;


    char *datos =
        coma
        +
        1;


    /* ====================================================
     * VALIDAR MODO
     * ==================================================== */

    int modo_features =
        (
            strcmp(
                modo,
                "FEATURES"
            )
            ==
            0
        );


    int modo_dnn =
        (
            strcmp(
                modo,
                "DNN"
            )
            ==
            0
        );


    int modo_kmeans =
        (
            strcmp(
                modo,
                "KMEANS"
            )
            ==
            0
        );


    int modo_all =
        (
            strcmp(
                modo,
                "ALL"
            )
            ==
            0
        );


    if (
        !modo_features
        &&
        !modo_dnn
        &&
        !modo_kmeans
        &&
        !modo_all
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "STACKBENCH_ERROR,MODO,%s\r\n",
            modo
        );

        return;
    }


    /* ====================================================
     * PARSEAR VENTANA RAW
     *
     * 16 muestras x 3 ejes = 48 valores
     * ==================================================== */

    float window[
        FE_WINDOW_SIZE
    ][
        FE_AXIS_COUNT
    ];


    int cantidad =
        0;


    char *token =
        strtok(
            datos,
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
         * El token completo debe representar
         * un número válido.
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
                "STACKBENCH_ERROR,VALOR,%d\r\n",
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
            "STACKBENCH_ERROR,CANTIDAD,%d\r\n",
            cantidad
        );

        return;
    }


    if (token != NULL)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "STACKBENCH_ERROR,EXTRA\r\n"
        );

        return;
    }


    /* ====================================================
     * VARIABLES DE TRABAJO
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


    KMeansResult resultado_kmeans;


    /* ====================================================
     * BASELINE DEL MONITOR
     *
     * Esto nos permite estimar la pequeña sobrecarga
     * introducida por la llamada mediante function pointer.
     * ==================================================== */

    StackMonitorResult resultado_base;


    int estado_monitor_base =
        StackMonitor_Measure(
            StackBench_RunBaseline,
            NULL,
            &resultado_base
        );


    if (estado_monitor_base != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "STACKBENCH_ERROR,BASE,%d\r\n",
            estado_monitor_base
        );

        return;
    }


    uint32_t stack_base =
        resultado_base.bytes_used;


    /* ====================================================
     * RESULTADO DE LA ETAPA
     * ==================================================== */

    StackMonitorResult resultado_stack;


    int estado_monitor =
        0;


    /* ====================================================
     * FEATURES
     * ==================================================== */

    if (modo_features)
    {
        StackBenchFeaturesContext ctx =
        {
            .window =
                window,

            .features =
                features
        };


        estado_monitor =
            StackMonitor_Measure(
                StackBench_RunFeatures,
                &ctx,
                &resultado_stack
            );
    }


    /* ====================================================
     * DNN
     *
     * Las features se calculan FUERA de la medición.
     * ==================================================== */

    else if (modo_dnn)
    {
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
                "STACKBENCH_ERROR,FEATURES,%d\r\n",
                estado_features
            );

            return;
        }


        StackBenchDNNContext ctx =
        {
            .features =
                features,

            .clase =
                &clase,

            .probs =
                probs
        };


        estado_monitor =
            StackMonitor_Measure(
                StackBench_RunDNN,
                &ctx,
                &resultado_stack
            );
    }


    /* ====================================================
     * KMEANS
     *
     * Las features se calculan FUERA de la medición.
     * ==================================================== */

    else if (modo_kmeans)
    {
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
                "STACKBENCH_ERROR,FEATURES,%d\r\n",
                estado_features
            );

            return;
        }


        StackBenchKMeansContext ctx =
        {
            .features =
                features,

            .resultado =
                &resultado_kmeans
        };


        estado_monitor =
            StackMonitor_Measure(
                StackBench_RunKMeans,
                &ctx,
                &resultado_stack
            );
    }


    /* ====================================================
     * PIPELINE COMPLETO
     * ==================================================== */

    else
    {
        StackBenchAllContext ctx =
        {
            .window =
                window,

            .features =
                features,

            .clase =
                &clase,

            .probs =
                probs,

            .resultado_kmeans =
                &resultado_kmeans
        };


        estado_monitor =
            StackMonitor_Measure(
                StackBench_RunAll,
                &ctx,
                &resultado_stack
            );
    }


    /* ====================================================
     * VALIDAR MONITOR
     * ==================================================== */

    if (estado_monitor != 0)
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "STACKBENCH_ERROR,MONITOR,%d\r\n",
            estado_monitor
        );

        return;
    }


    /* ====================================================
     * STACK AJUSTADO
     *
     * Eliminamos únicamente el baseline observado
     * por el mecanismo del monitor.
     *
     * Conservamos también el valor bruto en la salida
     * para mantener trazabilidad del experimento.
     * ==================================================== */

    uint32_t stack_raw =
        resultado_stack.bytes_used;


    uint32_t stack_ajustado =
        (
            stack_raw
            >
            stack_base
        )
        ?
        (
            stack_raw
            -
            stack_base
        )
        :
        0U;


    /* ====================================================
     * RESPUESTA
     *
     * STACKBENCH,
     * modo,
     * raw,
     * baseline,
     * ajustado,
     * status
     * ==================================================== */

    snprintf(
        respuesta,
        capacidad_respuesta,
        "STACKBENCH,%s,%lu,%lu,%lu,%d\r\n",
        modo,
        (unsigned long)stack_raw,
        (unsigned long)stack_base,
        (unsigned long)stack_ajustado,
        resultado_stack.function_status
    );
}

