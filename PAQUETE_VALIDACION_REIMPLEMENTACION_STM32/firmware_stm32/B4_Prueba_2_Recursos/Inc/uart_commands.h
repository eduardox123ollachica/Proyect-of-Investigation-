#ifndef UART_COMMANDS_H
#define UART_COMMANDS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Procesa un comando UART terminado en '\0' y escribe su respuesta.
 * El comando puede modificarse durante el procesamiento de PRED.
 */
void ProcesarComandoUART(
    char *comando,
    int estado_tinyml,
    char *respuesta,
    size_t capacidad_respuesta
);

#ifdef __cplusplus
}
#endif

#endif
