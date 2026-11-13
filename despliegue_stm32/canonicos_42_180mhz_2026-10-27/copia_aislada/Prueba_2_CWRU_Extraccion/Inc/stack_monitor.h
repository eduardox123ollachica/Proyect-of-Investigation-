#ifndef STACK_MONITOR_H
#define STACK_MONITOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Región máxima que se utiliza para detectar
 * el descenso del Main Stack Pointer (MSP).
 */
#define STACK_MONITOR_REGION_BYTES 4096U


typedef int (*StackMonitorFunction)(void *context);


/*
 * Resultado de una medición.
 */
typedef struct
{
    /*
     * Máximo descenso detectado desde el MSP
     * capturado antes de ejecutar la función.
     */
    uint32_t bytes_used;

    /*
     * Tamaño total de la región sentinel.
     */
    uint32_t region_bytes;

    /*
     * MSP justo antes de ejecutar la función.
     */
    uint32_t sp_before;

    /*
     * Dirección más baja modificada.
     */
    uint32_t lowest_address;

    /*
     * Valor devuelto por la función medida.
     */
    int function_status;

} StackMonitorResult;


/*
 * Ejecuta una función y determina su máximo
 * consumo dinámico de stack mediante watermark.
 *
 * Retorno:
 *
 *   0  -> medición válida
 *  -1  -> argumentos inválidos
 *  -2  -> espacio disponible insuficiente
 *  -3  -> la función alcanzó el límite inferior
 *         de la región monitorizada
 */
int StackMonitor_Measure(
    StackMonitorFunction function,
    void *context,
    StackMonitorResult *result
);


#ifdef __cplusplus
}
#endif

#endif
