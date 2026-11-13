/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>

extern int verificar_modelo_tflm(void);

void CmdModelo_Execute(char *respuesta, size_t capacidad_respuesta)
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

