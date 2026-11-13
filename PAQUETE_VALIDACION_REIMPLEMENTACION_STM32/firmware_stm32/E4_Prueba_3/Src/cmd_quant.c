/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include "tinyml_inference.h"

void CmdQuant_Execute(char *respuesta, size_t capacidad_respuesta)
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

