#include "stm32f4xx_hal.h"

#include <stdio.h>
#include <string.h>

#include "feature_extraction.h"
#include "tinyml_inference.h"
#include "kmeans_anomaly.h"


UART_HandleTypeDef huart2;


/* ============================================================
 * PROTOTIPOS
 * ============================================================ */

static void MX_GPIO_Init(void);

static void MX_USART2_UART_Init(void);

static void DWT_Init(void);

static void Error_Handler(void);


/* ============================================================
 * MAIN
 *
 * BUILD B4
 *
 * Incluye:
 *
 * - HAL
 * - GPIO
 * - UART2
 * - DWT
 * - SysTick
 * - Feature Extraction
 * - DNN TinyML
 * - K-means
 *
 * NO incluye:
 *
 * - Stack Monitor
 * - uart_commands.c
 * ============================================================ */

int main(void)
{
    /* ========================================================
     * HAL
     * ======================================================== */

    HAL_Init();


    /* ========================================================
     * DWT
     * ======================================================== */

    DWT_Init();


    /* ========================================================
     * PERIFÉRICOS
     * ======================================================== */

    MX_GPIO_Init();

    MX_USART2_UART_Init();


    /* ========================================================
     * B4 - PIPELINE COMPLETO
     *
     * Se fuerza la inclusión real de:
     *
     * 1. Feature Extraction
     * 2. TinyML / DNN
     * 3. K-means
     *
     * ======================================================== */

    float test_window[
        FE_WINDOW_SIZE
    ][
        FE_AXIS_COUNT
    ] =
    {
        {0.0f}
    };


    float test_features[
        FE_TOTAL_FEATURES
    ] =
    {
        0.0f
    };


    /* ========================================================
     * 1. FEATURE EXTRACTION
     * ======================================================== */

    int estado_features =
        FeatureExtraction_Extract(
            test_window,
            test_features
        );


    /* ========================================================
     * 2. TINYML INIT
     * ======================================================== */

    int estado_tinyml =
        TinyML_Init();


    int clase =
        -1;


    float probs[3] =
    {
        0.0f,
        0.0f,
        0.0f
    };


    /* ========================================================
     * 3. DNN
     * ======================================================== */

    int estado_pred =
        TinyML_Predict(
            test_features,
            &clase,
            probs
        );


    /* ========================================================
     * 4. K-MEANS
     * ======================================================== */

    KMeansResult resultado_kmeans;


    int estado_kmeans =
        KMeans_Compute(
            test_features,
            &resultado_kmeans
        );


    /* ========================================================
     * EVITAR WARNINGS
     * ======================================================== */

    (void)estado_features;
    (void)estado_tinyml;
    (void)estado_pred;
    (void)estado_kmeans;

    (void)clase;
    (void)probs;
    (void)resultado_kmeans;


    /* ========================================================
     * MENSAJE INICIAL
     * ======================================================== */

    const char inicio[] =
        "B4,READY\r\n";


    HAL_UART_Transmit(
        &huart2,
        (uint8_t *)inicio,
        strlen(inicio),
        HAL_MAX_DELAY
    );


    /* ========================================================
     * RECEPCIÓN UART
     *
     * Conservamos el mismo buffer para mantener comparable
     * la infraestructura con B0, B1, B2 y B3.
     * ======================================================== */

    uint8_t rx_buffer[1024];

    uint8_t rx_byte;

    uint32_t index = 0;


    /* ========================================================
     * LOOP PRINCIPAL
     * ======================================================== */

    while (1)
    {
        if (
            HAL_UART_Receive(
                &huart2,
                &rx_byte,
                1,
                10
            )
            ==
            HAL_OK
        )
        {
            /* =================================================
             * FIN DE COMANDO
             * ================================================= */

            if (
                rx_byte == '\n'
                ||
                rx_byte == '\r'
            )
            {
                if (index > 0)
                {
                    rx_buffer[index] =
                        '\0';


                    const char respuesta[] =
                        "B4,OK\r\n";


                    HAL_UART_Transmit(
                        &huart2,
                        (uint8_t *)respuesta,
                        strlen(respuesta),
                        HAL_MAX_DELAY
                    );


                    index = 0;
                }
            }

            /* =================================================
             * ALMACENAR BYTE
             * ================================================= */

            else
            {
                if (
                    index
                    <
                    sizeof(rx_buffer) - 1
                )
                {
                    rx_buffer[index++] =
                        rx_byte;
                }

                else
                {
                    index = 0;
                }
            }
        }
    }
}


/* ============================================================
 * DWT
 * ============================================================ */

static void DWT_Init(void)
{
    CoreDebug->DEMCR |=
        CoreDebug_DEMCR_TRCENA_Msk;


    DWT->CTRL &=
        ~DWT_CTRL_CYCCNTENA_Msk;


    DWT->CYCCNT =
        0;


    DWT->CTRL |=
        DWT_CTRL_CYCCNTENA_Msk;
}


/* ============================================================
 * GPIO
 *
 * PA2 -> USART2_TX
 * PA3 -> USART2_RX
 * ============================================================ */

static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct =
    {
        0
    };


    __HAL_RCC_GPIOA_CLK_ENABLE();


    GPIO_InitStruct.Pin =
        GPIO_PIN_2
        |
        GPIO_PIN_3;


    GPIO_InitStruct.Mode =
        GPIO_MODE_AF_PP;


    GPIO_InitStruct.Pull =
        GPIO_PULLUP;


    GPIO_InitStruct.Speed =
        GPIO_SPEED_FREQ_VERY_HIGH;


    GPIO_InitStruct.Alternate =
        GPIO_AF7_USART2;


    HAL_GPIO_Init(
        GPIOA,
        &GPIO_InitStruct
    );
}


/* ============================================================
 * USART2
 *
 * 115200 baud
 * 8 bits
 * 1 stop bit
 * sin paridad
 * ============================================================ */

static void MX_USART2_UART_Init(void)
{
    __HAL_RCC_USART2_CLK_ENABLE();


    huart2.Instance =
        USART2;


    huart2.Init.BaudRate =
        115200;


    huart2.Init.WordLength =
        UART_WORDLENGTH_8B;


    huart2.Init.StopBits =
        UART_STOPBITS_1;


    huart2.Init.Parity =
        UART_PARITY_NONE;


    huart2.Init.Mode =
        UART_MODE_TX_RX;


    huart2.Init.HwFlowCtl =
        UART_HWCONTROL_NONE;


    huart2.Init.OverSampling =
        UART_OVERSAMPLING_16;


    if (
        HAL_UART_Init(
            &huart2
        )
        !=
        HAL_OK
    )
    {
        Error_Handler();
    }
}


/* ============================================================
 * ERROR
 * ============================================================ */

static void Error_Handler(void)
{
    while (1)
    {
    }
}


/* ============================================================
 * SYSTICK
 * ============================================================ */

void SysTick_Handler(void)
{
    HAL_IncTick();
}
