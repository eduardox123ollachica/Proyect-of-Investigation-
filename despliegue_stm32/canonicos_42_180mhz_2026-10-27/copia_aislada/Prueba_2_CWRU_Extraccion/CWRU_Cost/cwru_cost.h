#ifndef CWRU_COST_H_
#define CWRU_COST_H_

#include <stddef.h>
#include "cwru_feature_extraction.h"

#if !CWRU_PROFILE_EXTRACTOR || !CWRU_PIPELINE_TEST
#error "CWRUCOST requiere el perfil existente y el pipeline integrado."
#endif
#if CWRU_ENTROPY_MODE != 2 || CWRU_CACHE_SPECTRAL_POWER != 1
#error "El contrato CWRUCOST version 1 corresponde al extractor EntM/cache."
#endif
#if CWRU_STACK_TEST
#error "El diagnostico de costos se ejecuta sin pintar el stack."
#endif

#ifdef __cplusplus
extern "C" {
#endif
void CWRUCOST_Command(char *command, char *response, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
