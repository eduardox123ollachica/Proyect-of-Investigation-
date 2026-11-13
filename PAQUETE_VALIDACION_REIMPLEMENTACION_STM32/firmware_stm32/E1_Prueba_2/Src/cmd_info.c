/* Modulo extraido de uart_commands.c sin cambios de logica funcional. */
#include "uart_command_handlers.h"
#include <stdio.h>
#include "tinyml_inference.h"

void CmdInfo_Execute(char *respuesta, size_t capacidad_respuesta)
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

