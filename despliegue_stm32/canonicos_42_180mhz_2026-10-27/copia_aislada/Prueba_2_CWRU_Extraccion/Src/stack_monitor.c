#include "stack_monitor.h"

#include <stddef.h>

#include "stm32f4xx.h"


/*
 * Símbolo definido por el linker.
 *
 * _end marca el final de:
 *
 *   .data
 *   .bss
 *   variables globales/estáticas
 *
 * En nuestro firmware:
 *
 * _end = 0x20004578
 */
extern uint8_t _end;


/* ============================================================
 * LEER MSP
 * ============================================================ */

static inline uint32_t StackMonitor_GetMSP(void)
{
    uint32_t sp;

    __asm volatile(
        "mrs %0, msp"
        : "=r"(sp)
    );

    return sp;
}


/* ============================================================
 * MEDICIÓN DE STACK
 * ============================================================ */

int StackMonitor_Measure(
    StackMonitorFunction function,
    void *context,
    StackMonitorResult *result
)
{
    if (
        function == NULL
        ||
        result == NULL
    )
    {
        return -1;
    }


    /*
     * Capturamos el MSP después de entrar
     * en StackMonitor_Measure().
     *
     * De esta forma el propio frame principal
     * del monitor NO se contabiliza como stack
     * de la etapa medida.
     */
    uint32_t sp_before =
        StackMonitor_GetMSP();


    /*
     * Fin de las variables globales/estáticas.
     */
    uint32_t ram_static_end =
        (uint32_t)&_end;


    /*
     * Margen adicional para evitar acercarnos
     * a la zona estática.
     */
    const uint32_t guard_bytes =
        256U;


    uint32_t minimum_allowed =
        ram_static_end
        +
        guard_bytes;


    /*
     * Verificar que exista suficiente espacio
     * entre _end y el MSP actual.
     */
    if (
        sp_before
        <=
        (
            minimum_allowed
            +
            STACK_MONITOR_REGION_BYTES
        )
    )
    {
        return -2;
    }


    /*
     * El stack del Cortex-M crece hacia
     * direcciones menores.
     *
     *
     *             direcciones altas
     *
     *             sp_before
     *                 |
     *                 v
     *       +----------------------+
     *       |                      |
     *       | región monitorizada  |
     *       |      4096 B          |
     *       |                      |
     *       +----------------------+
     *           region_start
     *
     *             direcciones bajas
     */
    uint32_t region_end =
        sp_before;


    uint32_t region_start =
        region_end
        -
        STACK_MONITOR_REGION_BYTES;


    /*
     * Dirección inicial como palabras de 32 bits.
     */
    uint32_t *memory =
        (uint32_t *)region_start;


    const uint32_t words =
        STACK_MONITOR_REGION_BYTES
        /
        sizeof(uint32_t);


    /*
     * Patrón sentinel.
     */
    const uint32_t pattern =
        0xA5A5A5A5U;


    /* ========================================================
     * LLENAR REGIÓN
     * ======================================================== */

    for (
        uint32_t i = 0U;
        i < words;
        i++
    )
    {
        memory[i] =
            pattern;
    }


    /*
     * Garantizar que todas las escrituras a RAM
     * finalizaron antes de comenzar la medición.
     */
    __DSB();
    __ISB();


    /* ========================================================
     * DESHABILITAR INTERRUPCIONES
     *
     * IMPORTANTE:
     *
     * Las interrupciones permanecen deshabilitadas
     * no solo durante la función medida, sino también
     * durante la inspección posterior del sentinel.
     *
     * Así evitamos que:
     *
     *   SysTick
     *   UART IRQ
     *   otras IRQ
     *
     * consuman stack y contaminen la medición.
     * ======================================================== */

    uint32_t primask =
        __get_PRIMASK();


    __disable_irq();


    __DSB();
    __ISB();


    /* ========================================================
     * EJECUTAR ETAPA
     * ======================================================== */

    int function_status =
        function(
            context
        );


    __DSB();
    __ISB();


    /* ========================================================
     * BUSCAR WATERMARK
     * ======================================================== */

    uint32_t first_modified =
        region_end;


    int found =
        0;


    /*
     * Buscamos desde la dirección más baja.
     *
     * Si una palabra conserva 0xA5A5A5A5,
     * esa parte nunca fue alcanzada por el stack.
     */
    for (
        uint32_t i = 0U;
        i < words;
        i++
    )
    {
        if (
            memory[i]
            !=
            pattern
        )
        {
            first_modified =
                region_start
                +
                (
                    i
                    *
                    sizeof(uint32_t)
                );


            found =
                1;


            break;
        }
    }


    uint32_t bytes_used =
        0U;


    if (found)
    {
        bytes_used =
            region_end
            -
            first_modified;
    }


    /*
     * Detectar si toda la región fue utilizada.
     *
     * Esto indicaría que 4096 B no fueron
     * suficientes para medir el pico.
     */
    int overflow_region =
        (
            memory[0]
            !=
            pattern
        )
        ?
        1
        :
        0;


    /* ========================================================
     * GUARDAR RESULTADOS
     * ======================================================== */

    result->bytes_used =
        bytes_used;


    result->region_bytes =
        STACK_MONITOR_REGION_BYTES;


    result->sp_before =
        sp_before;


    result->lowest_address =
        found
        ?
        first_modified
        :
        region_end;


    result->function_status =
        function_status;


    /* ========================================================
     * RESTAURAR INTERRUPCIONES
     *
     * Recién después de haber inspeccionado
     * completamente la región.
     * ======================================================== */

    __DSB();
    __ISB();


    if (primask == 0U)
    {
        __enable_irq();
    }


    if (overflow_region)
    {
        return -3;
    }


    return 0;
}
