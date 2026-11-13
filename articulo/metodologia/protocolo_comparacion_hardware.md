# Protocolo de comparación STM32-XIAO ESP32S3

## Niveles de comparación

1. **Especificaciones:** arquitectura, frecuencia máxima, memorias, DSP/FPU,
   conectividad y entorno de despliegue, usando documentación del fabricante.
2. **Resultados reportados:** cifras del artículo original y del presente trabajo,
   dejando explícito que el alcance y método de medición no son equivalentes.
3. **Benchmark cruzado:** solo podrá afirmarse una superioridad de rendimiento o
   energía si el mismo modelo y pipeline se ejecutan y miden en ambas plataformas.

## Comparación permitida con la evidencia actual

- Demostrar portabilidad funcional de Xtensa LX7 a ARM Cortex-M4F.
- Comparar recursos nominales y recursos consumidos por el firmware STM32.
- Mostrar cumplimiento del plazo de 80 ms en STM32.
- Discutir el papel de CMSIS-DSP en la FFT y del FPU/DSP del Cortex-M4F.
- Contrastar conectividad integrada del XIAO con el enfoque de validación UART.

## Afirmaciones que deben evitarse

- Que STM32 es más rápido que XIAO usando 6.31 ms frente a 25 ms sin igualar el
  alcance de medición.
- Que una frecuencia de reloj menor implica por sí sola mayor eficiencia energética.
- Que la prueba UART constituye adquisición autónoma o despliegue industrial.

## Comparación experimental futura opcional

Portar exactamente E4, las 21 características y K-means al XIAO ESP32S3; usar las
mismas 1332 ventanas, medir las mismas etapas y fijar frecuencia, compilador y nivel
de optimización. La energía exigiría medición eléctrica en ambas placas.
