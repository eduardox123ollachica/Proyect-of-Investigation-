#ifndef CWRU_PIPELINE_CONFIG_H_
#define CWRU_PIPELINE_CONFIG_H_

#define CWRU_SAMPLE_RATE_HZ 12000
#define CWRU_WINDOW_SAMPLES 4096
#define CWRU_HOP_SAMPLES 2048
#define CWRU_NUM_CHANNELS 2
#define CWRU_NUM_FEATURES_PER_CHANNEL 7
#define CWRU_TOTAL_FEATURES 14

/* Copia de prueba: no inicializar el modelo anterior. */
#define CWRU_EXTRACTION_ONLY 1
#define CWRU_UART_PROTOCOL_VERSION 2

// Orden de entrada: DE, FE. No representan ejes ortogonales X/Y/Z.
// La ruta embebida recibe o adquiere señales ya muestreadas a 12 kHz.

#endif  // CWRU_PIPELINE_CONFIG_H_
