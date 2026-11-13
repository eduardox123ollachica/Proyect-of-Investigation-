/* Protocolo2: recepción transaccional, checksum FNV1a y extracción aislada. */
#include "uart_command_handlers.h"
#include "cwru_feature_extraction.h"
#include "stm32f4xx_hal.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CWRU_PIPELINE_TEST
#define CWRU_PIPELINE_TEST 0
#endif
#if CWRU_PIPELINE_TEST
#include "cwru_pipeline.h"
#endif

#ifndef CWRU_TOTALV2_TEST
#define CWRU_TOTALV2_TEST 0
#endif
#if CWRU_TOTALV2_TEST
#include "cwru_total.h"
#endif

#ifndef CWRU_MASK_BENCH
#define CWRU_MASK_BENCH 0
#endif
#if CWRU_MASK_BENCH
#include "cwru_masks.h"
#endif

#define TOTAL (CWRU_WINDOW_SAMPLES * CWRU_NUM_CHANNELS)
#define FRAME_VALUES_MAX 48U
/* Nunca colocar estos 64 KiB de trabajo en el stack. */
static float32_t g_window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS];
static float32_t g_fft_input[CWRU_FE_SCRATCH_FLOATS];
static float32_t g_fft_output[CWRU_FE_SCRATCH_FLOATS];
static float32_t g_features[CWRU_TOTAL_FEATURES];
static CWRU_FeatureExtractor g_extractor;
static uint32_t g_received,g_checksum;
static uint8_t g_active,g_initialized;

#if CWRU_PROFILE_EXTRACTOR
static uint8_t g_profile_valid;
static uint32_t g_profile_elapsed;
#endif

static void error(char *out,size_t cap,const char *reason,long detail)
{
    (void)snprintf(out,cap,"CWRUFEAT_ERROR,%s,%ld\r\n",reason,detail);
}

#if CWRU_PROFILE_EXTRACTOR
static void profile_info(char *out,size_t cap)
{
    if (!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) { error(out,cap,"DWT_DISABLED",-1L);return; }
    uint32_t overhead=CWRU_FeatureExtraction_ProfileReadOverhead();
    (void)snprintf(out,cap,"CWRUFEAT_PROFILE_INFO,1,%u,%u,%lu,%lu\r\n",
        (unsigned)CWRU_PROFILE_BLOCKS,(unsigned)CWRU_NUM_CHANNELS,
        (unsigned long)HAL_RCC_GetHCLKFreq(),(unsigned long)overhead);
}

static void profile_result(char *out,size_t cap)
{
    int used;
    const CWRU_ExtractionProfile *profile;
    if (!g_profile_valid) { error(out,cap,"NO_PROFILE",-1L);return; }
    profile=CWRU_FeatureExtraction_GetProfile();
    used=snprintf(out,cap,"CWRUFEAT_PROFILE,1,%08lx,%lu",
        (unsigned long)g_checksum,(unsigned long)g_profile_elapsed);
    if (used<0||(size_t)used>=cap) { error(out,cap,"RESPONSE_SIZE",-1L);return; }
    for (uint32_t channel=0U;channel<CWRU_NUM_CHANNELS;++channel)
        for (uint32_t block=0U;block<CWRU_PROFILE_BLOCKS;++block)
        {
            int written=snprintf(out+used,cap-(size_t)used,",%lu",
                (unsigned long)profile->cycles[channel][block]);
            if (written<0||(size_t)written>=cap-(size_t)used)
            { error(out,cap,"RESPONSE_SIZE",-1L);return; }
            used+=written;
        }
    if (cap-(size_t)used<3U) { error(out,cap,"RESPONSE_SIZE",-1L);return; }
    (void)snprintf(out+used,cap-(size_t)used,"\r\n");
}
#endif

static void begin(char *out,size_t cap)
{
#if CWRU_TOTALV2_TEST
    CWRUTOTAL_Invalidate();
#endif
#if CWRU_MASK_BENCH
    CWRUMASK_Invalidate();
#endif
#if CWRU_PIPELINE_TEST
    CWRUPIPE_Invalidate();
#endif
#if CWRU_PROFILE_EXTRACTOR
    g_profile_valid=0U;
#endif
    if (!g_initialized)
    {
        int status=CWRU_FeatureExtraction_Init(&g_extractor);
        if (status!=CWRU_FE_OK) { error(out,cap,"INIT",status);return; }
        g_initialized=1U;
    }
    g_received=0U;g_checksum=2166136261U;g_active=1U;
    (void)snprintf(out,cap,"CWRUFEAT_READY,%lu\r\n",(unsigned long)TOTAL);
}

static void data(char *command,char *out,size_t cap)
{
#if CWRU_TOTALV2_TEST
    CWRUTOTAL_Invalidate();
#endif
#if CWRU_MASK_BENCH
    CWRUMASK_Invalidate();
#endif
#if CWRU_PIPELINE_TEST
    CWRUPIPE_Invalidate();
#endif
    float values[FRAME_VALUES_MAX]; /* Commit sólo tras validar toda la trama. */
    uint32_t count=0U;
    char *p=command+14,*end;
    unsigned long offset;
    if (!g_active) { error(out,cap,"NO_BEGIN",-1L);return; }
    if (*p<'0'||*p>'9') { error(out,cap,"BAD_OFFSET",-1L);return; }
    errno=0;offset=strtoul(p,&end,10);
    if (errno||*end!=','||offset!=g_received)
    { error(out,cap,"OFFSET",(long)g_received);return; }
    p=end+1;
    while (*p)
    {
        float value;
        if (count>=FRAME_VALUES_MAX) { error(out,cap,"FRAME_VALUES",count);return; }
        if (*p==',') { error(out,cap,"EMPTY_VALUE",count);return; }
        errno=0;value=strtof(p,&end);
        if (end==p||errno||!isfinite(value)||(*end!=','&&*end!='\0'))
        { error(out,cap,"BAD_VALUE",count);return; }
        values[count++]=value;
        if (*end=='\0') { p=end;break; }
        p=end+1;
        if (*p=='\0') { error(out,cap,"EMPTY_VALUE",count);return; }
    }
    if (!count||g_received+count>TOTAL) { error(out,cap,"VALUE_COUNT",count);return; }
    for (uint32_t i=0U;i<count;++i)
    {
        uint32_t sample=g_received/CWRU_NUM_CHANNELS;
        uint32_t channel=g_received%CWRU_NUM_CHANNELS;
        g_window[sample][channel]=values[i];
        const uint8_t *bytes=(const uint8_t *)&values[i];
        for (uint32_t j=0U;j<sizeof(float);++j) g_checksum=(g_checksum^bytes[j])*16777619U;
        ++g_received;
    }
    (void)snprintf(out,cap,"CWRUFEAT_ACK,%lu\r\n",(unsigned long)g_received);
}

static void run(char *out,size_t cap)
{
#if CWRU_TOTALV2_TEST
    CWRUTOTAL_Invalidate();
#endif
#if CWRU_MASK_BENCH
    CWRUMASK_Invalidate();
#endif
#if CWRU_PIPELINE_TEST
    CWRUPIPE_Invalidate();
#endif
    uint32_t start,elapsed;
    int status,used;
#if CWRU_PROFILE_EXTRACTOR
    g_profile_valid=0U;
#endif
    if (!g_active||g_received!=TOTAL) { error(out,cap,"INCOMPLETE",g_received);return; }
    if (!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) { error(out,cap,"DWT_DISABLED",-1L);return; }
    start=DWT->CYCCNT;
    status=CWRU_FeatureExtraction_Extract(&g_extractor,g_window,g_features,g_fft_input,g_fft_output);
    elapsed=DWT->CYCCNT-start;
    if (status!=CWRU_FE_OK) { error(out,cap,"EXTRACT",status);return; }
#if CWRU_PROFILE_EXTRACTOR
    g_profile_elapsed=elapsed;
    g_profile_valid=1U;
#endif
    used=snprintf(out,cap,"CWRUFEAT_RESULT,%lu",(unsigned long)elapsed);
    if (used<0||(size_t)used>=cap) { error(out,cap,"RESPONSE_SIZE",-1L);return; }
    for (uint32_t i=0U;i<CWRU_TOTAL_FEATURES;++i)
    {
        int written=snprintf(out+used,cap-(size_t)used,",%.9g",(double)g_features[i]);
        if (written<0||(size_t)written>=cap-(size_t)used) { error(out,cap,"RESPONSE_SIZE",i);return; }
        used+=written;
    }
    if (cap-(size_t)used<3U) { error(out,cap,"RESPONSE_SIZE",-1L);return; }
    (void)snprintf(out+used,cap-(size_t)used,"\r\n");
}

void CmdCwruFeat_Execute(char *command,char *out,size_t cap)
{
    if (!command||!out||!cap)return;
    if (!strcmp(command,"CWRUFEAT_INFO"))
        (void)snprintf(out,cap,"CWRUFEAT_INFO,%d,%d,%d,%d,%d,%lu,%u\r\n",
            CWRU_UART_PROTOCOL_VERSION,CWRU_SAMPLE_RATE_HZ,CWRU_WINDOW_SAMPLES,
            CWRU_NUM_CHANNELS,TOTAL,(unsigned long)HAL_RCC_GetHCLKFreq(),
            (DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)?1U:0U);
    else if (!strcmp(command,"CWRUFEAT_BEGIN"))begin(out,cap);
    else if (!strncmp(command,"CWRUFEAT_DATA,",14))data(command,out,cap);
    else if (!strcmp(command,"CWRUFEAT_CHECK"))
        (void)snprintf(out,cap,"CWRUFEAT_CHECK,%lu,%08lx\r\n",(unsigned long)g_received,(unsigned long)g_checksum);
    else if (!strcmp(command,"CWRUFEAT_RUN"))run(out,cap);
#if CWRU_PROFILE_EXTRACTOR
    else if (!strcmp(command,"CWRUFEAT_PROFILE_INFO"))profile_info(out,cap);
    else if (!strcmp(command,"CWRUFEAT_PROFILE"))profile_result(out,cap);
#endif
    else error(out,cap,"UNKNOWN",-1L);
}

#if CWRU_PIPELINE_TEST
#include "cwru_pipeline_commands.inc"
#endif

#ifndef CWRU_STACK_TEST
#define CWRU_STACK_TEST 0
#endif
#if CWRU_STACK_TEST
#include "cwru_stack_commands.inc"
#endif

#ifndef CWRU_COST_PROFILE
#define CWRU_COST_PROFILE 0
#endif
#if CWRU_COST_PROFILE
#include "cwru_cost_commands.inc"
#endif

#if CWRU_MASK_BENCH
#include "cwru_masks_commands.inc"
#endif

#if CWRU_TOTALV2_TEST
#include "cwru_total_commands.inc"
#endif
