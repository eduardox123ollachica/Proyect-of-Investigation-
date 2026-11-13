/* Se incluye después de cwru_pipeline_commands.inc, solo CWRU_STACK_TEST=1.
 * Reutiliza ventana, extractor y publicación existentes; no duplica FFT/arena. */
#include "cwru_stack.h"
static CWRUSTK_Result g_stack_result;
static uint32_t g_stack_id;
typedef struct {
    CWRU_FeatureExtractor *extractor;
    const float32_t (*window)[CWRU_NUM_CHANNELS];
    float32_t *input,*output;
    CWRUPIPE_Result *result;
} CWRUSTK_Context;
static CWRUSTK_Context g_stack_context;

static void stack_error(char *out,size_t cap,const char *reason,long detail)
{ (void)snprintf(out,cap,"CWRUSTK_ERROR,%s,%ld\r\n",reason,detail); }
static int stack_format(char *out,size_t cap,const char *format,...)
{
    va_list args;va_start(args,format);int n=vsnprintf(out,cap,format,args);va_end(args);
    if (n<0 || (size_t)n>=cap) { stack_error(out,cap,"RESPONSE_SIZE",-1);return 0; }
    return 1;
}
__attribute__((noinline)) static int stack_empty(void *context)
{ (void)context;return 0; }
__attribute__((naked,noinline)) static int stack_probe(void *context __attribute__((unused)))
{
    __asm volatile("sub sp,sp,#128\n"
                   "ldr r0,=0x13579bdf\n"
                   "str r0,[sp]\n"
                   "add sp,sp,#128\n"
                   "movs r0,#0\n"
                   "bx lr\n");
}
__attribute__((noinline)) static int stack_pipeline(void *pointer)
{
    CWRUSTK_Context *c=(CWRUSTK_Context *)pointer;
    if (__get_PRIMASK()||__get_BASEPRI()||__get_FAULTMASK()) return -100;
    int status=CWRUPIPE_Execute(c->extractor,c->window,c->input,c->output,c->result);
    if (__get_PRIMASK()||__get_BASEPRI()||__get_FAULTMASK()) return -101;
    return status;
}
static int stack_pattern(uint32_t word)
{ return word==0xa5a5a5a5U || word==0x5a5a5a5aU; }

static void stack_run(const char *args,char *out,size_t cap)
{
    CWRUPIPE_Invalidate();g_stack_id=0U;
    uint32_t expected,pattern;char token[9];
    if (strlen(args)!=17U || args[8]!=',') { stack_error(out,cap,"FORMAT",-1);return; }
    memcpy(token,args,8U);token[8]='\0';
    if (!pipe_checksum_word(token,&expected)||!pipe_checksum_word(args+9,&pattern)) { stack_error(out,cap,"FORMAT",-1);return; }
    if (!stack_pattern(pattern)) { stack_error(out,cap,"PATTERN",-1);return; }
    if (!g_active||g_received!=TOTAL) { stack_error(out,cap,"INCOMPLETE",g_received);return; }
    if (expected!=g_checksum) { stack_error(out,cap,"CHECKSUM",-1);return; }
    if (!g_initialized||CWRUNN_Status()!=0) { stack_error(out,cap,"NOT_INITIALIZED",CWRUNN_Status());return; }
    g_stack_context.extractor=&g_extractor;g_stack_context.window=g_window;
    g_stack_context.input=g_fft_input;g_stack_context.output=g_fft_output;g_stack_context.result=&g_pipe_result;
    memset(&g_stack_result,0,sizeof(g_stack_result));
    int status=CWRUSTK_Measure(stack_pipeline,&g_stack_context,pattern,&g_stack_result);
    if (status!=0) { stack_error(out,cap,"MONITOR",status);return; }
    if (g_stack_result.callback_status!=0) { stack_error(out,cap,"CALLBACK",g_stack_result.callback_status);return; }
    uint32_t next=g_pipe_id+1U;if (!next) next=1U;
    char words[CWRUSTK_WORDS*8U+1U];pipe_hex(words,&g_stack_result,sizeof(g_stack_result));
    if (stack_format(out,cap,"CWRUSTK_RESULT,%lu,%08lx,%08lx,%s\r\n",
        (unsigned long)next,(unsigned long)g_checksum,(unsigned long)pattern,words)) {
        g_pipe_id=next;g_pipe_checksum=g_checksum;g_pipe_valid=1U;g_stack_id=next;
    }
}

void CWRUSTK_Command(char *command,char *out,size_t cap)
{
    if (!command||!out||!cap) return;
    if (!strcmp(command,"CWRUSTK_INFO")) {
        extern uint8_t _estack,_end;extern uint32_t _Min_Stack_Size;
        stack_format(out,cap,"CWRUSTK_INFO,1,4096,128,34,%08lx,%08lx,%08lx,%08lx,%lu,%lu,%08lx,%lu,%08lx,%08lx\r\n",
            (unsigned long)&_estack,(unsigned long)((uint32_t)&_estack-(uint32_t)&_Min_Stack_Size),
            (unsigned long)&_end,(unsigned long)__get_CONTROL(),(unsigned long)__get_IPSR(),
            (unsigned long)__get_PRIMASK(),(unsigned long)__get_BASEPRI(),(unsigned long)__get_FAULTMASK(),
            (unsigned long)FPU->FPCCR,(unsigned long)SCB->CCR);return;
    }
    if (!strncmp(command,"CWRUSTK_INFO,",13)) { stack_error(out,cap,"FORMAT",-1);return; }
    if (!strcmp(command,"CWRUSTK_RUN")) { stack_run("",out,cap);return; }
    if (!strncmp(command,"CWRUSTK_RUN,",12)) { stack_run(command+12,out,cap);return; }
    if (!strcmp(command,"CWRUSTK_CAL") || !strncmp(command,"CWRUSTK_CAL,",12)) {
        CWRUPIPE_Invalidate();g_stack_id=0U;
        const char *args=command+strlen("CWRUSTK_CAL");
        if (*args!=',') { stack_error(out,cap,"FORMAT",-1);return; }
        ++args;const char *comma=strchr(args,',');
        if (!comma) { stack_error(out,cap,"FORMAT",-1);return; }
        CWRUSTK_Callback callback=NULL;const char *mode;
        if (comma-args==5 && !strncmp(args,"EMPTY",5)) { callback=stack_empty;mode="EMPTY"; }
        else if (comma-args==5 && !strncmp(args,"PROBE",5)) { callback=stack_probe;mode="PROBE"; }
        else { stack_error(out,cap,"MODE",-1);return; }
        uint32_t pattern;
        if (!pipe_checksum_word(comma+1,&pattern)) { stack_error(out,cap,"FORMAT",-1);return; }
        if (!stack_pattern(pattern)) { stack_error(out,cap,"PATTERN",-1);return; }
        memset(&g_stack_result,0,sizeof(g_stack_result));
        int status=CWRUSTK_Measure(callback,NULL,pattern,&g_stack_result);
        if (status!=0) { stack_error(out,cap,"MONITOR",status);return; }
        char words[CWRUSTK_WORDS*8U+1U];pipe_hex(words,&g_stack_result,sizeof(g_stack_result));
        stack_format(out,cap,"CWRUSTK_CAL,%s,%08lx,%s\r\n",mode,(unsigned long)pattern,words);return;
    }
    if (!strncmp(command,"CWRUSTK_TIMING,",15)) {
        uint32_t id;
        if (!pipe_record_id(command+15,&id)) { stack_error(out,cap,"FORMAT",-1);return; }
        if (!g_pipe_valid||!g_stack_id||g_stack_id!=g_pipe_id) { stack_error(out,cap,"NO_RESULT",-1);return; }
        if (id!=g_stack_id) { stack_error(out,cap,"STALE_ID",-1);return; }
        stack_format(out,cap,"CWRUPIPE_RESULT,%lu,%08lx,%lu,%lu,%lu,%lu,%lu,%lu,%d,%d,%d,%d,%08lx,%d,%lu,%lu\r\n",
            (unsigned long)g_pipe_id,(unsigned long)g_pipe_checksum,(unsigned long)g_pipe_result.total_cycles,
            (unsigned long)g_pipe_result.extract_cycles,(unsigned long)g_pipe_result.input_cycles,
            (unsigned long)g_pipe_result.nn_call_cycles,(unsigned long)g_pipe_result.nn_invoke_cycles,
            (unsigned long)g_pipe_result.detector_cycles,g_pipe_result.predicted_class,
            (int)g_pipe_result.output[0],(int)g_pipe_result.output[1],(int)g_pipe_result.output[2],
            (unsigned long)pipe_bits(g_pipe_result.detector.score),g_pipe_result.detector.rejected,
            (unsigned long)CWRUNN_ArenaUsed(),(unsigned long)g_pipe_result.extractions);return;
    }
    stack_error(out,cap,"UNKNOWN",-1);
}
