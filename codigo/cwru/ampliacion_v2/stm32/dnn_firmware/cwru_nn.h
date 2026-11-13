#ifndef CWRU_NN_H
#define CWRU_NN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CWRUNN_INPUTS 14U
#define CWRUNN_OUTPUTS 3U
#define CWRUNN_ARENA_CAPACITY 16384U
#define CWRUNN_MODEL_SHA256 "cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596"

int CWRUNN_Init(uint32_t *cycles, int *reused);
int CWRUNN_Status(void);
size_t CWRUNN_ArenaUsed(void);
uint32_t CWRUNN_ModelFNV(void);
int CWRUNN_PredictQ(const int8_t input[CWRUNN_INPUTS], int8_t output[CWRUNN_OUTPUTS],
                    int *predicted_class, uint32_t *invoke_cycles);
const char *CWRUNN_LastLog(void);
void CWRUNN_Command(char *command, char *response, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
