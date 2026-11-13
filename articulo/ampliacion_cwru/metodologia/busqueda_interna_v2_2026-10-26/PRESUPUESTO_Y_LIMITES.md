# Presupuesto congelado: ajustes DNN internos restantes

Este presupuesto prepara 240 ajustes nuevos de los 288 DNN internos. Se prevé
reutilizar 48 resultados completos por firma, incluidos 32 representantes
y 16 P0 del suplemento. La reutilización se valida por separado antes de ejecutar.
No se incluyen RF/SVM, reajustes externos, nuevos ajustes del detector ni hardware.
El piloto clásico ya está cerrado y no se repite.

## Evidencia temporal y previsión

Los 32 procesos recientes aprobados sumaron 450.845 s: media
14.089 s, mediana 15.146 s y máximo 16.579 s por ajuste,
incluyendo importación, entrenamiento, conversiones y cierre del proceso.
Las épocas fueron 11–50 (media 19.46875); solo seis llegaron
a 50. Se conservan máximo 50, paciencia 10, min_delta=1e-12 y restauración de
mejores pesos. No son 50 épocas fijas.

| Proyección para los 240 ajustes | Minutos |
|---|---:|
| Media de los procesos medidos × cantidad | 56.36 |
| Máximo de proceso observado × cantidad | 66.31 |
| Sensibilidad conservadora de componentes con 50 épocas | 95.40 |

Reservar aproximadamente **1–2 horas de trabajo restante**, repartido en lotes.
La sensibilidad a 50 épocas combina máximos de etapas de distintos procesos y
extrapola la peor razón observada de tiempo de ajuste por actualización a
1300 actualizaciones. Es una proyección de ingeniería, no una duración medida,
intervalo de confianza o garantía. Las comprobaciones previas y auditorías
pueden añadir tiempo. El arranque frío puede alargar los procesos y agotar
el presupuesto sin completar todos los ajustes.

La cifra anterior de unas 3 h 26 min tenía otro alcance: hasta 408 DNN,
328 RF, 100 SVM y 108 ajustes de detector antes de reutilización. No equivale
a esta etapa de DNN internas restantes.

## Límites y ejecución personal

- Primer lote: **un ajuste**, presupuesto de lote 600 s; revisar antes de ampliar.
- Lotes siguientes: como máximo **16 ajustes**, presupuesto de lote 1800 s.
- Cada proceso: máximo **300 s**, CPU con un hilo y ejecución secuencial.
- Presupuesto acumulado de procesos de la campaña: **7200 s**, persistente.

Con el perfil medido, 16 procesos sumarían unos
3.76 min; la
sensibilidad de componentes suma unos
6.36 min.
Conviene reservar **4–10 minutos por lote**, con techo solicitado de 30 minutos.
No iniciar el siguiente proceso si quedan menos de 300 s de lote o presupuesto
acumulado; detenerse en esa frontera conserva resultados completos. Los cinco
minutos por proceso no implican que cada ajuste vaya a tardar cinco minutos.

El presupuesto acumulado contabiliza procesos, no todo el tiempo de reloj:
verificación, preparación y auditoría pueden añadir tiempo fuera de ellos.
La terminación por timeout puede requerir una breve gracia adicional.
Reanudar no reinicia el presupuesto. No hay reintentos ni ampliación automática;
un fallo conserva el intento y detiene la cola para revisión.

El usuario ejecutará personalmente los comandos. Este documento y su generador
solo leen evidencias existentes: no entrenan, importan TensorFlow ni abren COM.
Los hashes de resultados, estados y cierres utilizados figuran en el JSON.
