#ifndef CWRU_CANON_POST_H_
#define CWRU_CANON_POST_H_
#include "cwru_total.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Features PC fijadas; no ejecuta FFT, extracción o adquisición. */
int CWRUCANON_PostExecute(uint32_t id,const float features[14],CWRUTOTAL_Result *result);
#ifdef __cplusplus
}
#endif
#endif
