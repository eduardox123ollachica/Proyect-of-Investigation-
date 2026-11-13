#ifndef CWRU_NORMALITY_H_
#define CWRU_NORMALITY_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CWRUND_PARAMS_SHA256 "d04f04476227f48b6dc489088de5793cdb4235aa6965adbb126e6ca020493808"
enum { CWRUND_RUN = 0, CWRUND_DIST = 1, CWRUND_CHECK = 2 };
typedef struct {
    float z[12];
    float terms[12];
    float score;
    int rejected;
} CWRUND_Result;
int CWRUND_Compute(const float *values, unsigned mode, CWRUND_Result *result);
uint32_t CWRUND_ParamsFNV(void);
void CWRUND_Command(char *command, char *out, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
