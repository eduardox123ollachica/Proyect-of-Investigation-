#include "cwru_total.h"
#include "cwru_total_nn.h"
#include "stm32f4xx_hal.h"
#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float)==4U && FLT_RADIX==2 && FLT_MANT_DIG==24,"IEEE float32 requerido");
_Static_assert(sizeof(int32_t)==4U && CWRU_WINDOW_SAMPLES==4096 && CWRU_NUM_CHANNELS==2,"Contrato incompatible");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "La serializacion de parametros requiere little endian"
#endif

static uint32_t hash_bytes(uint32_t hash,const void *p,size_t length)
{
    const unsigned char *bytes=(const unsigned char *)p;
    for(size_t i=0;i<length;++i) hash=(hash^bytes[i])*16777619U;
    return hash;
}

uint32_t CWRUTOTAL_ParamsFNV(const CWRUTV_Descriptor *d)
{
    if(d==NULL) return 0U;
    const uint8_t prefix[]={d->dims,d->det_dims,d->mask_id};
    uint32_t hash=hash_bytes(2166136261U,prefix,3U);
    hash=hash_bytes(hash,d->indices,14U);hash=hash_bytes(hash,d->det_indices,12U);
    hash=hash_bytes(hash,d->mean,56U);hash=hash_bytes(hash,d->scale,56U);
    hash=hash_bytes(hash,d->det_mean,48U);hash=hash_bytes(hash,d->det_scale,48U);
    hash=hash_bytes(hash,d->centroid,48U);hash=hash_bytes(hash,&d->threshold,4U);
    hash=hash_bytes(hash,&d->qin_scale,4U);return hash_bytes(hash,&d->qin_zero,4U);
}

int CWRUTOTAL_ValidateDescriptor(uint32_t id)
{
    if(id>=CWRUTV_COUNT) return -1;
    const CWRUTV_Descriptor *d=&cwru_total_catalog[id];
    const CWRUMASK_Spec *spec=CWRUMASK_GetSpec(d->mask_id);
    if(spec==NULL || d->mask_id>=8U || d->dims==0U || d->dims>14U || d->det_dims==0U || d->det_dims>12U ||
       d->model==NULL || d->model_size==0U || !isfinite(d->qin_scale) || d->qin_scale<=0.0f ||
       d->qin_zero< -128 || d->qin_zero>127 || !isfinite(d->threshold) || d->threshold<0.0f) return -2;
    uint16_t selected=0U,detected=0U;
    for(unsigned j=0;j<d->dims;++j) {
        unsigned i=d->indices[j];
        if(i>=14U || (selected&(1U<<i)) || !isfinite(d->mean[j]) || !isfinite(d->scale[j]) || d->scale[j]<=0.0f) return -3;
        selected|=(uint16_t)(1U<<i);
    }
    if(selected!=spec->feature_mask) return -4;
    for(unsigned j=0;j<d->det_dims;++j) {
        unsigned i=d->det_indices[j];
        if(i>=14U || i==4U || i==11U || (detected&(1U<<i)) || !(selected&(1U<<i)) ||
           !isfinite(d->det_mean[j]) || !isfinite(d->det_scale[j]) || d->det_scale[j]<=0.0f ||
           !isfinite(d->centroid[j])) return -5;
        detected|=(uint16_t)(1U<<i);
    }
    if(detected!=(uint16_t)(selected&~((1U<<4U)|(1U<<11U)))) return -6;
    return CWRUTOTAL_ParamsFNV(d)==d->params_fnv ? 0 : -7;
}

uint32_t CWRUTOTAL_ReadOverhead(void)
{
    static uint32_t minimum;
    if(!(DWT->CTRL&DWT_CTRL_CYCCNTENA_Msk)) return 0U;
    if(minimum==0U) {
        minimum=UINT32_MAX;
        for(unsigned i=0;i<64U;++i) {
            volatile uint32_t start=DWT->CYCCNT;
            volatile uint32_t end=DWT->CYCCNT;
            uint32_t delta=end-start;if(delta<minimum) minimum=delta;
        }
    }
    return minimum;
}

static int input_transform(const CWRUTV_Descriptor *d,CWRUTOTAL_Result *out)
{
    for(unsigned j=0;j<d->dims;++j) {
        float value=out->features[d->indices[j]];
        if(!isfinite(value)) return -10;
        volatile float centered=value-d->mean[j];
        float z=centered/d->scale[j];
        volatile float divided=z/d->qin_scale;
        float affine=divided+(float)d->qin_zero;
        if(!isfinite(centered) || !isfinite(z) || !isfinite(divided) || !isfinite(affine)) return -11;
        out->standardized[j]=z;out->affine[j]=affine;
        if(affine>=127.5f) { out->quantized[j]=127;out->saturated_high|=(uint16_t)(1U<<j); }
        else if(affine< -128.5f) { out->quantized[j]=-128;out->saturated_low|=(uint16_t)(1U<<j); }
        else {
            float base=floorf(affine);int rounded=(int)base;float fraction=affine-base;
            if(fraction>0.5f || (fraction==0.5f && rounded%2!=0)) ++rounded;
            out->quantized[j]=(int8_t)rounded;
        }
    }
    return 0;
}

static int normality(const CWRUTV_Descriptor *d,CWRUTOTAL_Result *out)
{
    float total=0.0f;
    for(unsigned j=0;j<d->det_dims;++j) {
        volatile float centered=out->features[d->det_indices[j]]-d->det_mean[j];
        float z=centered/d->det_scale[j];
        volatile float difference=z-d->centroid[j];
        volatile float square=difference*difference;
        volatile float next=total+square;
        if(!isfinite(centered) || !isfinite(z) || !isfinite(difference) || !isfinite(square) || !isfinite(next)) return -12;
        out->detector_z[j]=z;out->detector_terms[j]=square;total=next;
    }
    out->score=total;out->rejected=total>d->threshold;return 0;
}

int CWRUTOTAL_Execute(uint32_t id,unsigned profile,CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS],CWRUTOTAL_Result *out)
{
    if(id>=CWRUTV_COUNT || profile>1U || extractor==NULL || window==NULL || scratch_input==NULL ||
       scratch_output==NULL || out==NULL || CWRUTNN_SelectedId()!=id || CWRUTNN_Status()!=0) return -20;
    const CWRUTV_Descriptor *d=&cwru_total_catalog[id];
    memset(out,0,sizeof(*out)); /* Slots omitidos +0 y limpieza FUERA de ambas envolventes. */
    int status;uint32_t start,stage;
    if(profile==0U) {
        start=DWT->CYCCNT;
        ++out->extractions;
        status=CWRUMASK_Extract(d->mask_id,extractor,window,out->features,scratch_input,scratch_output);
        if(status!=0) { out->failed_stage=1U;return status; }
        status=input_transform(d,out);
        if(status!=0) { out->failed_stage=2U;return status; }
        status=CWRUTNN_Predict(out->quantized,d->dims,out->output,&out->predicted_class);
        if(status!=0) { out->failed_stage=3U;return status; }
        status=normality(d,out);
        if(status!=0) { out->failed_stage=4U;return status; }
        out->total_cycles=DWT->CYCCNT-start;
    } else {
        start=DWT->CYCCNT;stage=DWT->CYCCNT;
        ++out->extractions;
        status=CWRUMASK_Extract(d->mask_id,extractor,window,out->features,scratch_input,scratch_output);
        out->extract_cycles=DWT->CYCCNT-stage;
        if(status!=0) { out->failed_stage=1U;return status; }
        stage=DWT->CYCCNT;status=input_transform(d,out);out->input_cycles=DWT->CYCCNT-stage;
        if(status!=0) { out->failed_stage=2U;return status; }
        stage=DWT->CYCCNT;status=CWRUTNN_Predict(out->quantized,d->dims,out->output,&out->predicted_class);
        out->nn_cycles=DWT->CYCCNT-stage;
        if(status!=0) { out->failed_stage=3U;return status; }
        stage=DWT->CYCCNT;status=normality(d,out);out->detector_cycles=DWT->CYCCNT-stage;
        if(status!=0) { out->failed_stage=4U;return status; }
        out->total_cycles=DWT->CYCCNT-start;
        uint32_t sum=out->extract_cycles+out->input_cycles+out->nn_cycles+out->detector_cycles;
        if(sum>out->total_cycles) return -21;
        out->residual_cycles=out->total_cycles-sum;
    }
    return 0;
}

#include "cwru_canon_post_math.inc"
