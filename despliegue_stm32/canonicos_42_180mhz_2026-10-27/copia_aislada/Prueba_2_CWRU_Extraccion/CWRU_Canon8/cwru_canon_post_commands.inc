/* Incluir tras total_run y antes de CWRUTOTAL_Command en la copia privada. */
#include "cwru_canon_post.h"

static int canon_nibble(char c)
{
    if(c>='0' && c<='9') return c-'0';
    if(c>='a' && c<='f') return c-'a'+10;
    if(c>='A' && c<='F') return c-'A'+10;
    return -1;
}

static int CWRUCANON_Command(char *command,char *out,size_t cap)
{
    if(!strcmp(command,"CWRUTOTAL_POST_INFO")) {
        total_format(out,cap,"CWRUTOTAL_POST_INFO,1,14,56,0\r\n");return 1;
    }
    if(!strncmp(command,"CWRUTOTAL_POST_INFO,",19U)) {
        total_error(out,cap,"FORMAT",-1L);return 1;
    }
    const char *prefix="CWRUTOTAL_POST,";
    if(strncmp(command,prefix,strlen(prefix))) {
        if(!strcmp(command,"CWRUTOTAL_POST")) { CWRUTOTAL_Invalidate();total_error(out,cap,"FORMAT",-1L);return 1; }
        return 0;
    }
    CWRUTOTAL_Invalidate();
    const char *p=command+strlen(prefix);
    if(strlen(p)!=121U || p[8]!=',') { total_error(out,cap,"FORMAT",-1L);return 1; }
    char checksum_text[9];memcpy(checksum_text,p,8U);checksum_text[8]='\0';
    uint32_t expected;
    if(!total_checksum(checksum_text,&expected)) { total_error(out,cap,"FORMAT",-1L);return 1; }
    unsigned char bytes[56];uint32_t checksum=2166136261U;
    for(unsigned i=0;i<56U;++i) {
        int hi=canon_nibble(p[9U+2U*i]),lo=canon_nibble(p[10U+2U*i]);
        if(hi<0 || lo<0) { total_error(out,cap,"FORMAT",-1L);return 1; }
        bytes[i]=(unsigned char)((unsigned)hi*16U+(unsigned)lo);
        checksum=(checksum^bytes[i])*16777619U;
    }
    if(checksum!=expected) { total_error(out,cap,"CHECKSUM",-1L);return 1; }
    float features[14];memcpy(features,bytes,sizeof(features));
    for(unsigned i=0;i<14U;++i) if(!isfinite(features[i])) { total_error(out,cap,"NONFINITE",(long)i);return 1; }
    if(!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) { total_error(out,cap,"DWT_DISABLED",-1L);return 1; }
    if((__get_FPSCR()&((3U<<22U)|(1U<<24U)))!=0U) { total_error(out,cap,"FPSCR",-1L);return 1; }
    uint32_t id=CWRUTNN_SelectedId();
    if(id>=CWRUTV_COUNT || CWRUTNN_Status()!=0) { total_error(out,cap,"NO_SELECTION",-1L);return 1; }
    int status=CWRUCANON_PostExecute(id,features,&g_total_result);
    if(status!=0) { total_error(out,cap,"EXECUTE",status);return 1; }
    uint32_t next=g_total_id+1U;if(next==0U) next=1U;
    int formatted=total_format(out,cap,
        "CWRUTOTAL_POST_RESULT,%lu,%lu,%08lx,0,%lu,0,0,0,0,0,%d,%d,%d,%d,%08lx,%d,%lu,0\r\n",
        (unsigned long)next,(unsigned long)id,(unsigned long)checksum,
        (unsigned long)g_total_result.total_cycles,g_total_result.predicted_class,
        (int)g_total_result.output[0],(int)g_total_result.output[1],(int)g_total_result.output[2],
        (unsigned long)total_bits(g_total_result.score),g_total_result.rejected,(unsigned long)CWRUTNN_ArenaUsed());
    if(formatted) { g_total_id=next;g_total_representative=id;g_total_checksum=checksum;g_total_valid=1U; }
    return 1;
}
