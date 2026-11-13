/* Incluir al final de la copia privada de cwru_total.c. */
#include "cwru_canon_post.h"

int CWRUCANON_PostExecute(uint32_t id,const float features[14],CWRUTOTAL_Result *out)
{
    if(id>=CWRUTV_COUNT || features==NULL || out==NULL ||
       CWRUTNN_SelectedId()!=id || CWRUTNN_Status()!=0) return -40;
    const CWRUTV_Descriptor *d=&cwru_total_catalog[id];
    for(unsigned i=0;i<14U;++i) if(!isfinite(features[i])) return -41;
    memset(out,0,sizeof(*out));
    for(unsigned j=0;j<d->dims;++j) out->features[d->indices[j]]=features[d->indices[j]];
    const uint32_t start=DWT->CYCCNT;
    int status=input_transform(d,out);
    if(status!=0) { out->failed_stage=2U;return status; }
    status=CWRUTNN_Predict(out->quantized,d->dims,out->output,&out->predicted_class);
    if(status!=0) { out->failed_stage=3U;return status; }
    status=normality(d,out);
    if(status!=0) { out->failed_stage=4U;return status; }
    out->total_cycles=DWT->CYCCNT-start;
    /* extractions==0; total_cycles mide solo el procesamiento posterior.
     * Nunca se suma al costo RAW ni se presenta como pipeline completo. */
    return 0;
}
