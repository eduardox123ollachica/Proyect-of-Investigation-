#include "cwru_stack.h"
#include "stm32f4xx_hal.h"
extern uint8_t _estack,_end;
extern uint32_t _Min_Stack_Size,_Min_Heap_Size;
_Static_assert(sizeof(CWRUSTK_Result)==CWRUSTK_WORDS*4U,"Registro stack incompatible");

/* Revisar ensamblado de este monitor en cada build. Solo callback puede
 * consumir stack tras capturar MSP; paint/scan son accesos volátiles inline.
 * La huella escrita observada no equivale al mínimo SP exacto. */
__attribute__((noinline))
int CWRUSTK_Measure(CWRUSTK_Callback callback,void *context,uint32_t pattern,CWRUSTK_Result *r)
{
    if (!callback||!r) return -1;
    if (pattern!=0xa5a5a5a5U && pattern!=0x5a5a5a5aU) return -2;
    r->pattern=pattern;r->reserved=0U;r->monitor_status=-3;r->callback_status=-1;
    r->fpscr_before=__get_FPSCR();
    r->control_before=__get_CONTROL();r->ipsr_before=__get_IPSR();
    r->primask_before=__get_PRIMASK();r->basepri_before=__get_BASEPRI();r->faultmask_before=__get_FAULTMASK();
    r->fpccr_before=FPU->FPCCR;r->ccr_before=SCB->CCR;
    if ((r->control_before&3U)||r->ipsr_before||r->primask_before||r->basepri_before||r->faultmask_before) return -3;
    r->top=(uint32_t)&_estack;
    if ((uint32_t)&_Min_Stack_Size!=CWRUSTK_RESERVE) return -4;
    r->low=r->top-CWRUSTK_RESERVE;
    if (r->top!=0x20020000U || r->low!=0x2001f000U ||
        (uint32_t)&_end+(uint32_t)&_Min_Heap_Size>r->low) return -4;
    if (!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) return -5;
    r->ticks_before=HAL_GetTick(); /* Fuera de la región pintada. */
    uint32_t paint_start=DWT->CYCCNT;
    __disable_irq();
    __asm volatile("mrs %0, msp" : "=r"(r->reference_msp) :: "memory");
    uint32_t reference=r->reference_msp,low=r->low;
    if ((reference&7U)||reference>r->top||reference<=low+CWRUSTK_GUARD+256U) {
        __set_PRIMASK(r->primask_before);return -6;
    }
    r->scanned_bytes=reference-low;
    r->prefix_bytes=r->top-reference;
    for (volatile uint32_t *p=(volatile uint32_t *)low;p<(volatile uint32_t *)reference;++p) *p=pattern;
    __DSB();
    __set_PRIMASK(r->primask_before);
    r->paint_masked_cycles=DWT->CYCCNT-paint_start;
    __asm volatile("" ::: "memory");
    int status=callback(context); /* IRQ activas durante todo el pipeline. */
    uint32_t returned_primask=__get_PRIMASK();
    uint32_t scan_start=DWT->CYCCNT;
    __disable_irq();
    __asm volatile("mrs %0, msp" : "=r"(r->after_msp) :: "memory");
    /* returned_primask se capturó antes de enmascarar para el escaneo. */
    uint32_t first=reference,guard=1U;
    for (volatile uint32_t *p=(volatile uint32_t *)low;p<(volatile uint32_t *)reference;++p) {
        if (*p!=pattern) {
            if (first==reference) first=(uint32_t)p;
            if ((uint32_t)p<low+CWRUSTK_GUARD) guard=0U;
        }
    }
    r->lowest_modified=first;r->guard_intact=guard;
    r->additional_written_bytes=reference-first;
    r->total_observed_bytes=r->prefix_bytes+r->additional_written_bytes;
    r->callback_status=status;
    r->basepri_after=__get_BASEPRI();r->faultmask_after=__get_FAULTMASK();r->ipsr_after=__get_IPSR();
    r->control_after=__get_CONTROL();r->fpccr_after=FPU->FPCCR;r->ccr_after=SCB->CCR;
    r->fpscr_after=__get_FPSCR();
    __set_BASEPRI(r->basepri_before);
    __set_FAULTMASK(r->faultmask_before);
    __set_PRIMASK(r->primask_before);
    r->scan_masked_cycles=DWT->CYCCNT-scan_start;
    r->primask_after=returned_primask;
    r->ticks_after=HAL_GetTick(); /* Después del escaneo; no contamina watermark. */
    r->monitor_status=(!guard)?-7:(r->after_msp!=reference)?-8:
        (r->primask_after||r->basepri_after||r->faultmask_after||r->ipsr_after||(r->control_after&3U))?-9:0;
    return r->monitor_status;
}
