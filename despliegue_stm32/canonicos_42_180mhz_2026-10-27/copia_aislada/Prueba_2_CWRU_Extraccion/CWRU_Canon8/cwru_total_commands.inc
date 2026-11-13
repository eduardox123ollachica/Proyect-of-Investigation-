/* Incluir al final de cmd_cwru_feat.c: comparte receptor, ventana y FFT. */
#include "cwru_total.h"
#include "cwru_total_nn.h"
#include <stdarg.h>

static CWRUTOTAL_Result g_total_result;
static uint32_t g_total_id,g_total_checksum,g_total_representative;
static uint8_t g_total_valid;
void CWRUTOTAL_Invalidate(void) { g_total_valid=0U; }

static void total_error(char *out,size_t cap,const char *reason,long detail)
{ (void)snprintf(out,cap,"CWRUTOTAL_ERROR,%s,%ld\r\n",reason,detail); }

static int total_format(char *out,size_t cap,const char *format,...)
{
    va_list args;va_start(args,format);int n=vsnprintf(out,cap,format,args);va_end(args);
    if(n<0 || (size_t)n>=cap) { total_error(out,cap,"RESPONSE_SIZE",-1L);return 0; }
    return 1;
}

static uint32_t total_bits(float value) { uint32_t word;memcpy(&word,&value,4U);return word; }
static void total_hex(char *out,const void *pointer,unsigned count)
{
    const unsigned char *bytes=(const unsigned char *)pointer;const char *digits="0123456789abcdef";
    for(unsigned i=0;i<count;++i) { out[2U*i]=digits[bytes[i]>>4];out[2U*i+1U]=digits[bytes[i]&15U]; }
    out[2U*count]='\0';
}

static int total_uint(const char *p,uint32_t *value,int zero_allowed)
{
    if(*p<'0' || *p>'9') return 0;
    for(const char *q=p;*q;++q) if(*q<'0' || *q>'9') return 0;
    errno=0;char *end;unsigned long parsed=strtoul(p,&end,10);
    if(errno || *end || parsed>UINT32_MAX || (!zero_allowed && parsed==0U)) return 0;
    *value=(uint32_t)parsed;return 1;
}

static int total_checksum(const char *p,uint32_t *value)
{
    if(strlen(p)!=8U) return 0;
    uint32_t result=0U;
    for(unsigned i=0;i<8U;++i) {
        char c=p[i];unsigned digit;
        if(c>='0' && c<='9') digit=(unsigned)(c-'0');
        else if(c>='a' && c<='f') digit=(unsigned)(c-'a'+10);
        else if(c>='A' && c<='F') digit=(unsigned)(c-'A'+10);
        else return 0;
        result=(result<<4)|digit;
    }
    *value=result;return 1;
}

static void total_run(const char *p,unsigned profile,char *out,size_t cap)
{
    CWRUTOTAL_Invalidate();uint32_t expected;
    if(!total_checksum(p,&expected)) { total_error(out,cap,"FORMAT",-1L);return; }
    if(!g_active || g_received!=TOTAL) { total_error(out,cap,"INCOMPLETE",(long)g_received);return; }
    if(expected!=g_checksum) { total_error(out,cap,"CHECKSUM",-1L);return; }
    if(!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) { total_error(out,cap,"DWT_DISABLED",-1L);return; }
    if((__get_FPSCR()&((3U<<22U)|(1U<<24U)))!=0U) { total_error(out,cap,"FPSCR",-1L);return; }
    uint32_t id=CWRUTNN_SelectedId();
    if(id>=CWRUTV_COUNT || CWRUTNN_Status()!=0) { total_error(out,cap,"NO_SELECTION",-1L);return; }
    if(!g_initialized) { total_error(out,cap,"NOT_INITIALIZED",-1L);return; }
    int status=CWRUTOTAL_Execute(id,profile,&g_extractor,g_window,g_fft_input,g_fft_output,&g_total_result);
    if(status!=0) { total_error(out,cap,"EXECUTE",status);return; }
    uint32_t next=g_total_id+1U;if(next==0U) next=1U;
    int formatted=total_format(out,cap,
        "CWRUTOTAL_RESULT,%lu,%lu,%08lx,%u,%lu,%lu,%lu,%lu,%lu,%lu,%d,%d,%d,%d,%08lx,%d,%lu,%lu\r\n",
        (unsigned long)next,(unsigned long)id,(unsigned long)g_checksum,profile,
        (unsigned long)g_total_result.total_cycles,(unsigned long)g_total_result.extract_cycles,
        (unsigned long)g_total_result.input_cycles,(unsigned long)g_total_result.nn_cycles,
        (unsigned long)g_total_result.detector_cycles,(unsigned long)g_total_result.residual_cycles,
        g_total_result.predicted_class,(int)g_total_result.output[0],(int)g_total_result.output[1],(int)g_total_result.output[2],
        (unsigned long)total_bits(g_total_result.score),g_total_result.rejected,
        (unsigned long)CWRUTNN_ArenaUsed(),(unsigned long)g_total_result.extractions);
    if(formatted) { g_total_id=next;g_total_representative=id;g_total_checksum=g_checksum;g_total_valid=1U; }
}

#include "cwru_canon_post_commands.inc"

void CWRUTOTAL_Command(char *command,char *out,size_t cap)
{
    if(command==NULL || out==NULL || cap==0U) return;
    if(CWRUCANON_Command(command,out,cap)) return;
    if(!strcmp(command,"CWRUTOTAL_INFO")) {
        long selected=CWRUTNN_SelectedId()<CWRUTV_COUNT?(long)CWRUTNN_SelectedId():-1L;
        total_format(out,cap,"CWRUTOTAL_INFO,2,%u,12000,4096,2,%lu,%u,%08lx,%08lx,%s,%ld,%u,%lu,%lu,64,0\r\n",
            (unsigned)CWRUTV_COUNT,(unsigned long)HAL_RCC_GetHCLKFreq(),(unsigned)((DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)!=0U),
            (unsigned long)__get_FPSCR(),(unsigned long)cwru_total_catalog_fnv,cwru_total_catalog_sha256,
            selected,(unsigned)g_total_valid,(unsigned long)g_total_id,(unsigned long)CWRUTOTAL_ReadOverhead());return;
    }
    if(!strncmp(command,"CWRUTOTAL_INFO,",15)) { total_error(out,cap,"FORMAT",-1L);return; }
    const char *select_prefix="CWRUTOTAL_SELECT,";const char *model_prefix="CWRUTOTAL_MODEL,";
    int select=!strncmp(command,select_prefix,strlen(select_prefix));
    int model=!strncmp(command,model_prefix,strlen(model_prefix));
    if(select || model) {
        if(select) CWRUTOTAL_Invalidate();
        uint32_t id;const char *p=command+strlen(select?select_prefix:model_prefix);
        if(!total_uint(p,&id,1)) { total_error(out,cap,"FORMAT",-1L);return; }
        if(id>=CWRUTV_COUNT) { total_error(out,cap,"REP_ID",-1L);return; }
        const CWRUTV_Descriptor *d=&cwru_total_catalog[id];
        if(model) {
            total_format(out,cap,"CWRUTOTAL_MODEL,%lu,%u,%u,%u,%u,%08lx,%08lx,%08lx,%ld,%08lx,%s,%s\r\n",
                (unsigned long)id,(unsigned)d->mask_id,(unsigned)d->dims,(unsigned)d->det_dims,d->model_size,
                (unsigned long)d->model_fnv,(unsigned long)d->params_fnv,(unsigned long)total_bits(d->qin_scale),
                (long)d->qin_zero,(unsigned long)total_bits(d->threshold),d->model_sha256,d->source_fit_signature);return;
        }
        int valid=CWRUTOTAL_ValidateDescriptor(id);
        if(valid!=0) { total_error(out,cap,"PARAMS",valid);return; }
        uint32_t cycles;int reused;int status=CWRUTNN_Select(id,&cycles,&reused);
        if(status!=0) { total_error(out,cap,"INIT",status);return; }
        total_format(out,cap,"CWRUTOTAL_SELECTED,%lu,%lu,%lu,%d,%d,%08lx,%08lx\r\n",
            (unsigned long)id,(unsigned long)cycles,(unsigned long)CWRUTNN_ArenaUsed(),status,reused,
            (unsigned long)CWRUTNN_ActualModelFNV(),(unsigned long)CWRUTOTAL_ParamsFNV(d));return;
    }
    if(!strcmp(command,"CWRUTOTAL_RUN")) { total_run("",0U,out,cap);return; }
    if(!strcmp(command,"CWRUTOTAL_PROFILE")) { total_run("",1U,out,cap);return; }
    if(!strncmp(command,"CWRUTOTAL_RUN,",14)) { total_run(command+14,0U,out,cap);return; }
    if(!strncmp(command,"CWRUTOTAL_PROFILE,",18)) { total_run(command+18,1U,out,cap);return; }
    const char *prefixes[]={"CWRUTOTAL_FEATURES,","CWRUTOTAL_INPUT,","CWRUTOTAL_DETECTOR,"};
    unsigned page;const char *p=NULL;
    for(page=0;page<3U;++page) { size_t n=strlen(prefixes[page]);if(!strncmp(command,prefixes[page],n)) { p=command+n;break; } }
    if(p==NULL) { total_error(out,cap,"UNKNOWN",-1L);return; }
    uint32_t id;
    if(!total_uint(p,&id,0)) { total_error(out,cap,"FORMAT",-1L);return; }
    if(!g_total_valid) { total_error(out,cap,"NO_RESULT",-1L);return; }
    if(id!=g_total_id) { total_error(out,cap,"STALE_ID",-1L);return; }
    const CWRUTV_Descriptor *d=&cwru_total_catalog[g_total_representative];char a[113],b[113],q[29];
    if(page==0U) {
        total_hex(a,g_total_result.features,56U);
        total_format(out,cap,"CWRUTOTAL_FEATURES,%lu,%lu,%08lx,%s\r\n",(unsigned long)id,
            (unsigned long)g_total_representative,(unsigned long)g_total_checksum,a);
    } else if(page==1U) {
        total_hex(a,g_total_result.standardized,4U*d->dims);total_hex(b,g_total_result.affine,4U*d->dims);
        total_hex(q,g_total_result.quantized,d->dims);
        total_format(out,cap,"CWRUTOTAL_INPUT,%lu,%lu,%08lx,%s,%s,%s,%04x,%04x\r\n",(unsigned long)id,
            (unsigned long)g_total_representative,(unsigned long)g_total_checksum,a,b,q,
            (unsigned)g_total_result.saturated_low,(unsigned)g_total_result.saturated_high);
    } else {
        total_hex(a,g_total_result.detector_z,4U*d->det_dims);total_hex(b,g_total_result.detector_terms,4U*d->det_dims);
        total_format(out,cap,"CWRUTOTAL_DETECTOR,%lu,%lu,%08lx,%s,%s,%08lx,%d\r\n",(unsigned long)id,
            (unsigned long)g_total_representative,(unsigned long)g_total_checksum,a,b,
            (unsigned long)total_bits(g_total_result.score),g_total_result.rejected);
    }
}
