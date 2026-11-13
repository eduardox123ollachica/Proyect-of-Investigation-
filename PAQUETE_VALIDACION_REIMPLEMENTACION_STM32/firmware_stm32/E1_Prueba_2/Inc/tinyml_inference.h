#ifndef TINYML_INFERENCE_H
#define TINYML_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

int TinyML_Init(void);

int TinyML_InputType(void);
int TinyML_OutputType(void);

int TinyML_InputBytes(void);
int TinyML_OutputBytes(void);

float TinyML_InputScale(void);
int TinyML_InputZeroPoint(void);

float TinyML_OutputScale(void);
int TinyML_OutputZeroPoint(void);

/*
 * NUEVO:
 * Ejecuta una inferencia usando 21 características float.
 *
 * features -> vector de entrada de 21 características
 * clase    -> clase predicha: 0, 1 o 2
 * probs    -> 3 salidas desquantizadas
 */
int TinyML_Predict(
    const float features[21],
    int *clase,
    float probs[3]
);

#ifdef __cplusplus
}
#endif

#endif
