#ifndef CWRU_INPUT_TRANSFORM_H
#define CWRU_INPUT_TRANSFORM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CWRU_INPUT_FEATURES 14U
#define CWRU_INPUT_PROTOCOL_VERSION 1U
#define CWRU_INPUT_PARAMS_SHA256 "24275e363e29573e7a158d44b00099d2224be9a783ac87654f2e1575fc86f1a5"

typedef struct {
    float standardized[CWRU_INPUT_FEATURES];
    float affine[CWRU_INPUT_FEATURES];
    int8_t quantized[CWRU_INPUT_FEATURES];
    uint16_t saturated_low;
    uint16_t saturated_high;
} CWRU_InputResult;

enum {
    CWRU_INPUT_OK = 0,
    CWRU_INPUT_NULL = -1,
    CWRU_INPUT_NONFINITE = -2,
    CWRU_INPUT_INTERMEDIATE = -3
};

/* Contrato: float32 en cada resta/division/suma; ties-to-even DESPUES de ZP.
 * Ninguna funcion modifica la entrada. La salida sólo es válida con retorno OK.
 * Los modos QUANT y ROUND son diagnósticos, no etapas adicionales del pipeline.
 */
int CWRU_InputTransform(const float input[CWRU_INPUT_FEATURES], CWRU_InputResult *out);
int CWRU_InputQuantize(const float standardized[CWRU_INPUT_FEATURES], CWRU_InputResult *out);
int CWRU_InputRound(const float affine[CWRU_INPUT_FEATURES], CWRU_InputResult *out);
void CWRU_InputCommand(char *command, char *response, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
