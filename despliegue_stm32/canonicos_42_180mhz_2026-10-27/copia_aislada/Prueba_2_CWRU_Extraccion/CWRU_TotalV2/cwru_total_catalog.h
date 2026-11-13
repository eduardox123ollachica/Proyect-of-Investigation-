#ifndef CWRU_TOTAL_CATALOG_H_
#define CWRU_TOTAL_CATALOG_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CWRUTV_COUNT 32U
typedef struct {
 const char *id;
 const unsigned char *model;
 unsigned int model_size;
 uint8_t mask_id,dims,det_dims;
 uint8_t indices[14],det_indices[12];
 float mean[14],scale[14],det_mean[12],det_scale[12],centroid[12];
 float threshold,qin_scale;
 int32_t qin_zero;
 uint32_t model_fnv,params_fnv;
 const char *model_sha256,*source_fit_signature;
} CWRUTV_Descriptor;
extern const CWRUTV_Descriptor cwru_total_catalog[CWRUTV_COUNT];
extern const char cwru_total_catalog_sha256[];
extern const uint32_t cwru_total_catalog_fnv;
#ifdef __cplusplus
}
#endif
#endif
