#ifndef UART_COMMANDS_H
#define UART_COMMANDS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ProcesarComandoUART(
    char *comando,
    int estado_tinyml,
    char *respuesta,
    size_t capacidad_respuesta
);

#ifdef __cplusplus
}
#endif

#endif /* UART_COMMANDS_H */
