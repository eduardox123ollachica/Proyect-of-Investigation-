/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include "stm32f4xx_hal.h"

void CmdBenchInfo_Execute(char *respuesta, size_t capacidad_respuesta)
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

