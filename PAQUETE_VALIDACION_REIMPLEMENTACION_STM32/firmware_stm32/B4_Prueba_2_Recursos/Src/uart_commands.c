#include "uart_commands.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stm32f4xx_hal.h"

#include "feature_extraction.h"
#include "tinyml_inference.h"
#include "kmeans_anomaly.h"
#include "stack_monitor.h"

extern int verificar_modelo_tflm(void);

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

/* ============================================================
 * PROCESAMIENTO DE COMANDOS UART
 * ============================================================ */

void ProcesarComandoUART(
    char *comando,
    int estado_tinyml,
    char *respuesta,
    size_t capacidad_respuesta
)
{
    /* ========================================================
     * MODELO
     * ======================================================== */

    if (strcmp(comando, "MODELO") == 0)
    {
        int estado_modelo =
            verificar_modelo_tflm();

        snprintf(
            respuesta,
            capacidad_respuesta,
            "MODELO,%d\r\n",
            estado_modelo
        );
    }


    /* ========================================================
     * KMEANSRAW
     *
     * Entrada:
     *
     * KMEANSRAW,
     * ax0,ay0,az0,
     * ...
     * ax15,ay15,az15
     *
     * Total:
     *
     * 16 muestras x 3 ejes = 48 valores.
     *
     * Pipeline:
     *
     * ventana 16x3
     *      |
     *      v
     * FeatureExtraction_Extract()
     *      |
     *      v
     * 21 features
     *      |
     *      v
     * StandardScaler
     *      |
     *      v
     * K-means K=32
     *      |
     *      v
     * cluster + score_d2 + anomaly
     *
     * Respuesta:
     *
     * KMEANSRAW,cluster,score_d2,anomaly
     *
     * Ejemplo:
     *
     * KMEANSRAW,9,28.164337158,1
     *
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "KMEANSRAW,",
            10
        ) == 0
    )
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

    /* ========================================================
     * RAWALL
     *
     * Pipeline combinado:
     *
     * ventana RAW 16x3
     *        |
     *        v
     * FeatureExtraction_Extract()
     *        |
     *        v
     *   21 features
     *      /     \
     *     /       \
     *    v         v
     * TinyML     K-means
     *    |         |
     * clase      cluster
     * probs      score_d2
     *            anomaly
     *
     * Entrada:
     *
     * RAWALL,
     * ax0,ay0,az0,
     * ...
     * ax15,ay15,az15
     *
     * Respuesta:
     *
     * RAWALL,
     * clase,p0,p1,p2,
     * cluster,score_d2,anomaly
     *
     * Ejemplo:
     *
     * RAWALL,0,0.996094,0.003906,0.000000,29,68.296837,1
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "RAWALL,",
            7
        ) == 0
    )
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

    /* ========================================================
     * BENCHALL
     *
     * Pipeline medido:
     *
     * RAW 16x3
     *   ↓
     * FeatureExtraction_Extract()
     *   ↓
     * 21 features
     *   ├── TinyML_Predict()
     *   └── KMeans_Compute()
     *
     * Respuesta:
     *
     * BENCHALL,
     * clase,
     * anomaly,
     * ciclos_features,
     * ciclos_dnn,
     * ciclos_kmeans,
     * ciclos_total
     *
     * IMPORTANTE:
     *
     * NO se mide:
     * - recepción UART
     * - parsing ASCII
     * - strtof()
     * - snprintf()
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "BENCHALL,",
            9
        ) == 0
    )
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

    /* ========================================================
     * STACKBENCH
     *
     * Medición dinámica de stack mediante watermark.
     *
     * Modos:
     *
     * STACKBENCH,FEATURES,<48 valores>
     * STACKBENCH,DNN,<48 valores>
     * STACKBENCH,KMEANS,<48 valores>
     * STACKBENCH,ALL,<48 valores>
     *
     *
     * El parsing UART ocurre ANTES de iniciar el monitor.
     *
     *
     * Para DNN y KMEANS:
     *
     * FeatureExtraction_Extract() se ejecuta antes de iniciar
     * StackMonitor, de modo que su consumo NO forma parte
     * de la medición de esas etapas.
     *
     *
     * Para ALL:
     *
     * se mide:
     *
     * RAW
     *  |
     *  v
     * FEATURES
     *  |
     *  +------ DNN
     *  |
     *  +------ KMEANS
     *
     *
     * Respuesta:
     *
     * STACKBENCH,
     * MODO,
     * bytes_medidos,
     * baseline,
     * bytes_ajustados,
     * estado_funcion
     *
     *
     * Ejemplo:
     *
     * STACKBENCH,FEATURES,824,16,808,0
     *
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "STACKBENCH,",
            11
        ) == 0
    )
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

    /* ========================================================
     * TINYML
     * ======================================================== */

    else if (
        strcmp(
            comando,
            "TINYML"
        ) == 0
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "TINYML,%d\r\n",
            estado_tinyml
        );
    }


    /* ========================================================
     * INFO
     * ======================================================== */

    else if (
        strcmp(
            comando,
            "INFO"
        ) == 0
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "INFO,%d,%d,%d,%d\r\n",
            TinyML_InputType(),
            TinyML_OutputType(),
            TinyML_InputBytes(),
            TinyML_OutputBytes()
        );
    }


    /* ========================================================
     * QUANT
     * ======================================================== */

    else if (
        strcmp(
            comando,
            "QUANT"
        ) == 0
    )
    {
        snprintf(
            respuesta,
            capacidad_respuesta,
            "QUANT,%.9f,%d,%.9f,%d\r\n",
            TinyML_InputScale(),
            TinyML_InputZeroPoint(),
            TinyML_OutputScale(),
            TinyML_OutputZeroPoint()
        );
    }


    /* ========================================================
     * BENCHINFO
     *
     * Respuesta:
     *
     * BENCHINFO,HCLK_HZ,DWT_ENABLED
     *
     * Ejemplo:
     *
     * BENCHINFO,16000000,1
     *
     * ======================================================== */

    else if (
        strcmp(
            comando,
            "BENCHINFO"
        ) == 0
    )
    {
        uint32_t hclk_hz =
            HAL_RCC_GetHCLKFreq();


        uint32_t dwt_ctrl =
            DWT->CTRL;


        int dwt_enabled =
            (
                dwt_ctrl
                &
                DWT_CTRL_CYCCNTENA_Msk
            )
            ?
            1
            :
            0;


        snprintf(
            respuesta,
            capacidad_respuesta,
            "BENCHINFO,%lu,%d\r\n",
            (unsigned long)hclk_hz,
            dwt_enabled
        );
    }


    /* ========================================================
     * RAWFEAT
     *
     * Entrada:
     *
     * RAWFEAT,
     * ax0,ay0,az0,
     * ...
     * ax15,ay15,az15
     *
     * Salida:
     *
     * FEATURES,f01,...,f21
     *
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "RAWFEAT,",
            8
        ) == 0
    )
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


    /* ========================================================
     * RAWPRED
     *
     * Pipeline:
     *
     * ventana 16x3
     *      |
     *      v
     * FeatureExtraction_Extract()
     *      |
     *      v
     * 21 features
     *      |
     *      v
     * TinyML_Predict()
     *      |
     *      v
     * clase + probabilidades
     *
     * Respuesta:
     *
     * RAWPRED,clase,p0,p1,p2
     *
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "RAWPRED,",
            8
        ) == 0
    )
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


    /* ========================================================
     * BENCHRAW
     *
     * EXPERIMENTO 4
     *
     * Entrada:
     *
     * BENCHRAW,
     * ax0,ay0,az0,
     * ...
     * ax15,ay15,az15
     *
     * Total:
     *
     * 48 valores.
     *
     * IMPORTANTE:
     *
     * El parsing UART ocurre ANTES de iniciar DWT.
     *
     * Por tanto las mediciones NO incluyen:
     *
     * - recepción UART
     * - parsing de ASCII
     * - strtof()
     * - generación de respuesta
     *
     * Se mide únicamente:
     *
     * 1. FeatureExtraction_Extract()
     * 2. TinyML_Predict()
     * 3. Pipeline completo features + inference
     *
     * Respuesta:
     *
     * BENCHRAW,
     * clase,
     * ciclos_features,
     * ciclos_inferencia,
     * ciclos_total
     *
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "BENCHRAW,",
            9
        ) == 0
    )
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


    /* ========================================================
     * PRED
     *
     * Entrada:
     *
     * PRED,f01,...,f21
     *
     * Respuesta:
     *
     * PRED,clase,p0,p1,p2
     *
     * ======================================================== */

    else if (
        strncmp(
            comando,
            "PRED,",
            5
        ) == 0
    )
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


    /* ========================================================
     * COMANDO GENÉRICO
     * ======================================================== */

    else
    {
        int valor =
            atoi(
                comando
            );


        snprintf(
            respuesta,
            capacidad_respuesta,
            "RECIBIDO,%d\r\n",
            valor
        );
    }
}
