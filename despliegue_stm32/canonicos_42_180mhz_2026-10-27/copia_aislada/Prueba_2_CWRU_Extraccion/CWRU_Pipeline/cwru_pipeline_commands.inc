/* Incluir sólo al final de cmd_cwru_feat.c bajo CWRU_PIPELINE_TEST=1.
 * Reutiliza sus buffers y receptor transaccional; NO reserva otra ventana/FFT. */
#include "cwru_pipeline.h"
#include "cwru_normality_params.h"
#include <stdarg.h>

static CWRUPIPE_Result g_pipe_result;
static uint32_t g_pipe_id;
static uint32_t g_pipe_checksum;
static uint8_t g_pipe_valid;

void CWRUPIPE_Invalidate(void) { g_pipe_valid=0U; }

static void pipe_error(char *out,size_t cap,const char *reason,long detail)
{
    (void)snprintf(out,cap,"CWRUPIPE_ERROR,%s,%ld\r\n",reason,detail);
}

static int pipe_format(char *out,size_t cap,const char *format,...)
{
    va_list args;va_start(args,format);
    int n=vsnprintf(out,cap,format,args);va_end(args);
    if (n<0 || (size_t)n>=cap) { pipe_error(out,cap,"RESPONSE_SIZE",-1L);return 0; }
    return 1;
}

static uint32_t pipe_bits(float value) { uint32_t word;memcpy(&word,&value,4U);return word; }
static void pipe_hex(char *out,const void *pointer,unsigned count)
{
    const unsigned char *p=(const unsigned char *)pointer;
    const char digits[]="0123456789abcdef";
    for (unsigned i=0;i<count;++i) { out[2*i]=digits[p[i]>>4];out[2*i+1]=digits[p[i]&15U]; }
    out[2*count]='\0';
}

static int pipe_checksum_word(const char *p,uint32_t *value)
{
    if (strlen(p)!=8U) return 0;
    uint32_t word=0U;
    for (unsigned i=0;i<8U;++i) {
        char c=p[i];unsigned nibble;
        if (c>='0'&&c<='9') nibble=(unsigned)(c-'0');
        else if (c>='a'&&c<='f') nibble=(unsigned)(c-'a'+10);
        else if (c>='A'&&c<='F') nibble=(unsigned)(c-'A'+10);
        else return 0;
        word=(word<<4)|nibble;
    }
    *value=word;return 1;
}

static int pipe_record_id(const char *p,uint32_t *value)
{
    if (*p<'0'||*p>'9') return 0;
    const char *q=p;
    while (*q) { if (*q<'0'||*q>'9') return 0;++q; }
    char *end;errno=0;
    unsigned long parsed=strtoul(p,&end,10);
    if (errno || *end!='\0' || parsed==0UL || parsed>UINT32_MAX) return 0;
    *value=(uint32_t)parsed;return 1;
}

static void pipe_run(const char *p,char *out,size_t cap)
{
    CWRUPIPE_Invalidate();
    uint32_t expected;
    if (!pipe_checksum_word(p,&expected)) { pipe_error(out,cap,"FORMAT",-1L);return; }
    if (!g_active || g_received!=TOTAL) { pipe_error(out,cap,"INCOMPLETE",(long)g_received);return; }
    if (expected!=g_checksum) { pipe_error(out,cap,"CHECKSUM",-1L);return; }
    if (!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) { pipe_error(out,cap,"DWT_DISABLED",-1L);return; }
    if (!g_initialized || CWRUNN_Status()!=0) { pipe_error(out,cap,"NOT_INITIALIZED",CWRUNN_Status());return; }
    int status=CWRUPIPE_Execute(&g_extractor,g_window,g_fft_input,g_fft_output,&g_pipe_result);
    if (status!=0) {
        const char *names[]={"EXECUTE","EXTRACT","INPUT","DNN","DETECTOR"};
        unsigned stage=g_pipe_result.failed_stage;
        pipe_error(out,cap,stage<=4U?names[stage]:"EXECUTE",status);return;
    }
    uint32_t next_id=g_pipe_id+1U;if (next_id==0U) next_id=1U;
    int formatted=pipe_format(out,cap,"CWRUPIPE_RESULT,%lu,%08lx,%lu,%lu,%lu,%lu,%lu,%lu,%d,%d,%d,%d,%08lx,%d,%lu,%lu\r\n",
        (unsigned long)next_id,(unsigned long)g_checksum,(unsigned long)g_pipe_result.total_cycles,
        (unsigned long)g_pipe_result.extract_cycles,(unsigned long)g_pipe_result.input_cycles,
        (unsigned long)g_pipe_result.nn_call_cycles,(unsigned long)g_pipe_result.nn_invoke_cycles,
        (unsigned long)g_pipe_result.detector_cycles,g_pipe_result.predicted_class,
        (int)g_pipe_result.output[0],(int)g_pipe_result.output[1],(int)g_pipe_result.output[2],
        (unsigned long)pipe_bits(g_pipe_result.detector.score),g_pipe_result.detector.rejected,
        (unsigned long)CWRUNN_ArenaUsed(),(unsigned long)g_pipe_result.extractions);
    if (formatted) { g_pipe_id=next_id;g_pipe_checksum=g_checksum;g_pipe_valid=1U; }
}

void CWRUPIPE_Command(char *command,char *out,size_t cap)
{
    if (command==NULL||out==NULL||cap==0U) return;
    if (!strcmp(command,"CWRUPIPE_INFO")) {
        pipe_format(out,cap,"CWRUPIPE_INFO,1,12000,4096,2,14,%lu,%u,%08lx,4064,%08lx,%s,%s,%08lx,%08lx,16384,%lu,%d,%u,%lu\r\n",
            (unsigned long)HAL_RCC_GetHCLKFreq(),(unsigned)((DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)!=0U),
            (unsigned long)__get_FPSCR(),(unsigned long)CWRUNN_ModelFNV(),CWRUNN_MODEL_SHA256,
            CWRU_INPUT_PARAMS_SHA256,(unsigned long)CWRUND_ParamsFNV(),(unsigned long)pipe_bits(cwru_normality_threshold_d2),
            (unsigned long)CWRUNN_ArenaUsed(),CWRUNN_Status(),(unsigned)g_pipe_valid,(unsigned long)g_pipe_id);
        return;
    }
    if (!strcmp(command,"CWRUPIPE_INIT")) {
        if (!g_initialized) {
            int s=CWRU_FeatureExtraction_Init(&g_extractor);
            if (s!=0) { pipe_error(out,cap,"INIT",s);return; }
            g_initialized=1U;
        }
        uint32_t cycles;int reused;int status=CWRUNN_Init(&cycles,&reused);
        if (status!=0) { pipe_error(out,cap,"INIT",status);return; }
        pipe_format(out,cap,"CWRUPIPE_READY,%lu,%lu,%d,1\r\n",(unsigned long)cycles,
            (unsigned long)CWRUNN_ArenaUsed(),reused);return;
    }
    if (!strncmp(command,"CWRUPIPE_INFO,",14) || !strncmp(command,"CWRUPIPE_INIT,",14)) {
        pipe_error(out,cap,"FORMAT",-1L);return;
    }
    if (!strcmp(command,"CWRUPIPE_RUN")) { pipe_run("",out,cap);return; }
    if (!strncmp(command,"CWRUPIPE_RUN,",13)) { pipe_run(command+13,out,cap);return; }
    const char *prefixes[]={"CWRUPIPE_FEATURES,","CWRUPIPE_INPUT,","CWRUPIPE_DETECTOR,"};
    unsigned page;const char *p=NULL;
    for (page=0;page<3U;++page) {
        size_t n=strlen(prefixes[page]);
        if (!strncmp(command,prefixes[page],n)) { p=command+n;break; }
    }
    if (p==NULL) { pipe_error(out,cap,"UNKNOWN",-1L);return; }
    uint32_t id;
    if (!pipe_record_id(p,&id)) { pipe_error(out,cap,"FORMAT",-1L);return; }
    if (!g_pipe_valid) { pipe_error(out,cap,"NO_RESULT",-1L);return; }
    if (id!=g_pipe_id) { pipe_error(out,cap,"STALE_ID",-1L);return; }
    char a[113],b[113],q[29];
    if (page==0U) {
        pipe_hex(a,g_pipe_result.features,56U);
        pipe_format(out,cap,"CWRUPIPE_FEATURES,%lu,%08lx,%s\r\n",(unsigned long)id,(unsigned long)g_pipe_checksum,a);
    } else if (page==1U) {
        pipe_hex(a,g_pipe_result.input.standardized,56U);pipe_hex(b,g_pipe_result.input.affine,56U);
        pipe_hex(q,g_pipe_result.input.quantized,14U);
        pipe_format(out,cap,"CWRUPIPE_INPUT,%lu,%08lx,%s,%s,%s,%04x,%04x\r\n",
            (unsigned long)id,(unsigned long)g_pipe_checksum,a,b,q,
            (unsigned)g_pipe_result.input.saturated_low,(unsigned)g_pipe_result.input.saturated_high);
    } else {
        pipe_hex(a,g_pipe_result.detector.z,48U);pipe_hex(b,g_pipe_result.detector.terms,48U);
        pipe_format(out,cap,"CWRUPIPE_DETECTOR,%lu,%08lx,%s,%s,%08lx,%d,0\r\n",
            (unsigned long)id,(unsigned long)g_pipe_checksum,a,b,
            (unsigned long)pipe_bits(g_pipe_result.detector.score),g_pipe_result.detector.rejected);
    }
}
