#include "stm32f4xx_hal.h"

#include <stdio.h>
#include <string.h>

#include "tinyml_inference.h"
#include "uart_commands.h"


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
 * ============================================================ */

int main(void)
{
    /*
     * Inicialización HAL.
     */
    HAL_Init();


    /*
     * Inicializar contador de ciclos DWT.
     *
     * Se utilizará posteriormente para medir:
     *
     * - extracción de características
     * - inferencia TinyML
     * - pipeline completo
     */
    DWT_Init();


    /*
     * Inicialización de periféricos.
     */
    MX_GPIO_Init();

    MX_USART2_UART_Init();


    /*
     * Inicializa TensorFlow Lite Micro una sola vez.
     */
    int estado_tinyml =
        TinyML_Init();


    /*
     * Mensaje inicial.
     */
    char mensaje_tinyml[64];

    snprintf(
        mensaje_tinyml,
        sizeof(mensaje_tinyml),
        "TINYML,%d\r\n",
        estado_tinyml
    );


    HAL_UART_Transmit(
        &huart2,
        (uint8_t *)mensaje_tinyml,
        strlen(mensaje_tinyml),
        HAL_MAX_DELAY
    );


    /*
     * Variables para recepción UART.
     *
     * 1024 bytes permiten recibir los comandos
     * RAWFEAT y RAWPRED con 48 valores float.
     */
    uint8_t rx_buffer[1024];

    uint8_t rx_byte;

    uint32_t index = 0;


    const char inicio[] =
        "STM32 listo para recibir datos\r\n";


    HAL_UART_Transmit(
        &huart2,
        (uint8_t *)inicio,
        strlen(inicio),
        HAL_MAX_DELAY
    );


    /* ========================================================
     * LOOP PRINCIPAL
     * ======================================================== */

    while (1)
    {
        /*
         * Recibir un byte por UART.
         */
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
            /*
             * Si llega fin de línea,
             * procesamos el comando.
             */
            if (
                rx_byte == '\n'
                ||
                rx_byte == '\r'
            )
            {
                if (index > 0)
                {
                    /*
                     * Convertir buffer a cadena C.
                     */
                    rx_buffer[index] =
                        '\0';


                    /*
                     * Buffer de respuesta.
                     */
                    char respuesta[512];


                    /*
                     * Procesar comando recibido.
                     */
                    ProcesarComandoUART(
                        (char *)rx_buffer,
                        estado_tinyml,
                        respuesta,
                        sizeof(respuesta)
                    );


                    /*
                     * Enviar respuesta al PC.
                     */
                    HAL_UART_Transmit(
                        &huart2,
                        (uint8_t *)respuesta,
                        strlen(respuesta),
                        HAL_MAX_DELAY
                    );


                    /*
                     * Reiniciar el buffer.
                     */
                    index = 0;
                }
            }
            else
            {
                /*
                 * Guardar byte recibido.
                 */
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
                    /*
                     * Protección contra desbordamiento.
                     *
                     * Si se supera el tamaño del buffer,
                     * descartamos la trama actual.
                     */
                    index = 0;
                }
            }
        }
    }
}


/* ============================================================
 * INICIALIZACIÓN DWT
 * ============================================================ */

/*
 * Habilita el contador de ciclos del Cortex-M4.
 *
 * DWT->CYCCNT incrementa una vez por cada ciclo
 * del núcleo.
 *
 * Este contador permitirá realizar mediciones
 * de rendimiento sin incluir el tiempo UART.
 */
static void DWT_Init(void)
{
    /*
     * Habilitar acceso a los registros
     * de trace/debug.
     */
    CoreDebug->DEMCR |=
        CoreDebug_DEMCR_TRCENA_Msk;


    /*
     * Detener temporalmente el contador.
     */
    DWT->CTRL &=
        ~DWT_CTRL_CYCCNTENA_Msk;


    /*
     * Reiniciar contador.
     */
    DWT->CYCCNT =
        0;


    /*
     * Habilitar contador de ciclos.
     */
    DWT->CTRL |=
        DWT_CTRL_CYCCNTENA_Msk;
}


/* ============================================================
 * CONFIGURACIÓN GPIO
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
 * CONFIGURACIÓN USART2
 *
 * 115200 baudios
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
 * ERROR GENERAL
 * ============================================================ */

static void Error_Handler(void)
{
    while (1)
    {
    }
}


/* ============================================================
 * INTERRUPCIÓN SYSTICK
 * ============================================================ */

/*
 * Incremento del tick HAL.
 */
void SysTick_Handler(void)
{
    HAL_IncTick();
}
