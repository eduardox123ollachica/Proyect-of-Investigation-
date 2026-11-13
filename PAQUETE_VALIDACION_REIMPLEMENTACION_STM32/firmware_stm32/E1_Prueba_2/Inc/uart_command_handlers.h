#ifndef UART_COMMAND_HANDLERS_H
#define UART_COMMAND_HANDLERS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void CmdModelo_Execute(char *respuesta, size_t capacidad_respuesta);
void CmdKMeansRaw_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdRawAll_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdBenchAll_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdStackBench_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdTinyML_Execute(int estado_tinyml, char *respuesta, size_t capacidad_respuesta);
void CmdInfo_Execute(char *respuesta, size_t capacidad_respuesta);
void CmdQuant_Execute(char *respuesta, size_t capacidad_respuesta);
void CmdBenchInfo_Execute(char *respuesta, size_t capacidad_respuesta);
void CmdRawFeat_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdRawPred_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdBenchRaw_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);
void CmdPred_Execute(char *comando, char *respuesta, size_t capacidad_respuesta);

#ifdef __cplusplus
}
#endif

#endif /* UART_COMMAND_HANDLERS_H */
