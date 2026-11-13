#ifndef CWRU_TOTAL_NN_H_
#define CWRU_TOTAL_NN_H_

#include <stddef.h>
#include <stdint.h>

#define CWRUTNN_ARENA_CAPACITY 16384U
#define CWRUTNN_OUTPUTS 3U
#define CWRUTNN_NO_MODEL UINT32_MAX

#ifdef __cplusplus
extern "C" {
#endif

/* Select/AllocateTensors are initialization work, outside the timed RUN.
 * Re-selecting the ready model returns reused=1 and cycles=0. Any failed
 * selection invalidates the previous model, including an invalid ID.
 */
int CWRUTNN_Select(uint32_t id, uint32_t *cycles, int *reused);

/* Copies input, invokes the selected model, copies its three INT8 outputs,
 * and selects the lowest class index on ties. No DWT reads occur here.
 */
int CWRUTNN_Predict(const int8_t *input, uint8_t count,
    int8_t output[CWRUTNN_OUTPUTS], int *predicted_class);

int CWRUTNN_Status(void);
size_t CWRUTNN_ArenaUsed(void);
uint32_t CWRUTNN_SelectedId(void);

/* Computes FNV-1a over the selected model bytes, rather than returning the
 * descriptor's declared fingerprint. Call outside timed RUN. Returns 0
 * when no model is ready.
 */
uint32_t CWRUTNN_ActualModelFNV(void);

#ifdef __cplusplus
}
#endif

#endif /* CWRU_TOTAL_NN_H_ */
