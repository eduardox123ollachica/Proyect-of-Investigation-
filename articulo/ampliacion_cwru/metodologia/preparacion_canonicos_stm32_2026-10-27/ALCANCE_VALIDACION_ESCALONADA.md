# Preparación de la validación canónica PC–STM32

El 27 de octubre de 2026 el usuario eligió la validación escalonada. Se
mantienen los ocho modelos canónicos de semilla 42 y las decisiones del
cierre exterior. Esta preparación no incorpora resultados físicos nuevos
al manuscrito ni revisa la selección a partir de la prueba exterior.

La entrega se encuentra en
`despliegue_stm32/canonicos_42_180mhz_2026-10-27`. Su
[guía de ejecución y criterios](../../../../despliegue_stm32/canonicos_42_180mhz_2026-10-27/README_VALIDACION_ESCALONADA.md)
describe la configuración, los modelos, comandos y límites.

La cobertura se divide de esta forma:

| Entrada al STM32 | Cobertura | Qué permite comprobar |
|---|---|---|
| RAW, 4096 muestras DE/FE | 56 ventanas únicas; 42 de desarrollo por modelo; 336 pares | Extractor parcial, pipeline desde memoria y tiempos canónicos |
| Características PC guardadas | 1380 ventanas exteriores; dos modelos por carga; 2760 pares | Escalado, cuantización, DNN y detector para esas entradas |

El muestreo RAW toma la primera y última ventana de cada MAT. Cada modelo
excluye su carga exterior del banco RAW; la evaluación física de las
características de esa carga corresponde a POST. No se podrá afirmar una
validación RAW exhaustiva del conjunto exterior. Los dos bancos son
disjuntos para cada modelo, aunque una misma ventana puede tener otro
papel para un modelo con distinta carga exterior.

Se reutilizan los pesos y parámetros finales y las predicciones guardadas.
La preparación no entrena ni vuelve a invocar clasificadores PC. Las
comprobaciones aritméticas de las referencias y los hashes son pruebas de
integridad, no nuevos experimentos predictivos.

La ruta POST se añadió únicamente a una copia aislada del firmware; recibe
14 valores binary32 y declara cero extracciones. Se conserva la ruta RAW,
el extractor y la corrección single-rounding ya validados. El catálogo se
reduce de 32 representantes internos a ocho modelos canónicos, sin
convertir o modificar sus modelos INT8. Se añade una configuración y se
conservan las 17 anteriores.

La compilación completa corresponde al usuario. Tras ella se debe revisar
el ELF, su identidad, modelos y parámetros enlazados, kernels y memoria;
después, realizar gates de comunicación/equivalencia antes del lote físico.
La preparación, la compilación sintáctica y la auditoría estática no
demuestran ejecución correcta sobre la placa. El tiempo POST, el perfilado,
la transmisión UART y la inicialización no deben confundirse con el RUN
principal del pipeline RAW.

La generalización exterior conserva sus límites: estudio retrospectivo,
un banco de un mismo motor, sensibilidad a la carga y a la semilla, y
rechazo observado de registros normales. Esta etapa permite verificar la
implementación física y medir sus costos; no elimina esos resultados.
