#ifndef CWRU_STACK_DIAGNOSTIC_H_
#define CWRU_STACK_DIAGNOSTIC_H_
#include <stddef.h>
#include <stdint.h>
#define CWRUSTK_WORDS 34U
#define CWRUSTK_RESERVE 4096U
#define CWRUSTK_GUARD 128U
typedef int (*CWRUSTK_Callback)(void *context);
typedef struct {
    uint32_t pattern,top,low,reference_msp,after_msp,lowest_modified;
    uint32_t prefix_bytes,additional_written_bytes,total_observed_bytes,scanned_bytes,guard_intact;
    int32_t callback_status;
    uint32_t primask_before,primask_after,basepri_before,basepri_after,faultmask_before,faultmask_after;
    uint32_t control_before,control_after,fpccr_before,fpccr_after,ccr_before,ccr_after;
    uint32_t ticks_before,ticks_after,paint_masked_cycles,scan_masked_cycles,ipsr_before,ipsr_after;
    int32_t monitor_status;
    uint32_t reserved,fpscr_before,fpscr_after;
} CWRUSTK_Result;
int CWRUSTK_Measure(CWRUSTK_Callback callback,void *context,uint32_t pattern,CWRUSTK_Result *result);
void CWRUSTK_Command(char *command,char *response,size_t capacity);
#endif
