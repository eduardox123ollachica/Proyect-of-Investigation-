/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>

void CmdTinyML_Execute(int estado_tinyml, char *respuesta, size_t capacidad_respuesta)
{
    snprintf(
        respuesta,
        capacidad_respuesta,
        "TINYML,%d\r\n",
        estado_tinyml
    );
}

