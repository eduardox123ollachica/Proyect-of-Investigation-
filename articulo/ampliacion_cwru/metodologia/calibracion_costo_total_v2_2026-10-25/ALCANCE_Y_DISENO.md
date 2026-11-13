# Calibración del costo total V2: alcance antes de la prueba física

Fecha de preparación: 25 de octubre de 2026. Estado: recursos exportados y
código en preparación; compilación completa y medidas físicas pendientes.
Este documento es evidencia de método, no un resultado incorporado al
manuscrito.

## Continuidad del protocolo

Se mantiene el objetivo de costo total elegido por el usuario y las reglas
existentes de `protocolo_v2_2026-10-25`. El suplemento cerrado no promueve
P1 ni MOM4; esta calibración emplea los 32 representantes P0 ya entrenados,
con semilla 11 y firma de ajuste original. Son un subconjunto previsto de la
búsqueda interna de 288 ajustes, no una campaña nueva de entrenamiento.

Cada escenario exterior emplea como representante la menor carga de
validación interna de sus tres cargas de desarrollo. Su banco temporal usa
solo las otras dos cargas de entrenamiento: primera y última ventana de
cada uno de los catorce MAT, 28 ventanas. Las ocho máscaras comparten banco,
orden y número de repeticiones dentro de ese escenario.

| Carga exterior excluida (HP) | Validación interna (HP) | Cargas del banco temporal (HP) |
|---|---|---|
| 0 | 1 | 2, 3 |
| 1 | 0 | 2, 3 |
| 2 | 0 | 1, 3 |
| 3 | 0 | 1, 2 |

La unión contiene 42 ventanas únicas de cargas 1, 2 y 3. Veintiocho tienen
una referencia física EntM previa y catorce de 3 HP requieren su primera
comprobación física de extracción. Que 3 HP aparezca en los bancos de los
escenarios exteriores 0, 1 y 2 corresponde a sus roles de entrenamiento;
no se incorpora al banco del escenario exterior 3. La calibración temporal
no usa sus resultados predictivos para escoger ventanas.

La evaluación continúa siendo retrospectiva: los archivos ya se exploraron
durante el desarrollo. Este control de roles no acredita independencia de
motor o rodamiento ni convierte el conjunto en una prueba prospectiva.

## Unidad medida y volumen

Un RUN principal ejecuta una extracción especializada, escalado y
cuantización de la DNN, copia/Invoke/copia/argmax y detector de normalidad. Una
pareja de lecturas DWT envuelve todo el recorrido. La limpieza del resultado
y los slots omitidos se realizan fuera de la envolvente.

Hay 32 modelos por 28 ventanas por tres repeticiones: 2688 RUN principales,
84 observaciones por representante. Un PROFILE adicional por pareja añade
896 recorridos con lecturas de etapas y residual, excluidos del selector.
Se añaden 42 BASE EntM fuera de ambos grupos para la referencia numérica.
No hay extracciones redundantes dentro del RUN principal.

Se excluyen transmisión UART, parser, verificación de checksums, selección e
inicialización del modelo. Las lecturas DWT consecutivas se calibran con
64 parejas como diagnóstico, sin restar su mínimo a los tiempos medidos.
La frecuencia nominal HAL de 180 MHz se registra; todavía no equivale a
una medición externa del reloj.

## Verificación numérica

1. BASE EntM se contrasta con las 14 características PC: tolerancia relativa
   0.001, absoluta 0.00001 y bins dominantes exactos. Para las 28 ventanas
   físicas previas se exige además identidad de bits con la referencia sellada.
2. Los slots seleccionados de cada kernel deben ser idénticos a BASE; los
   inactivos deben ser cero positivo.
3. El replay PC desde las características publicadas por ARM reproduce, con
   operaciones float32 separadas, escalado, afinidad, redondeo al par, INT8 y
   distancia del detector. La salida TFLite debe coincidir exactamente para
   la misma entrada INT8.
4. Se registra por separado toda diferencia frente a las características,
   entradas INT8, salidas y decisiones del caché PC original. No se deduce
   equivalencia desde ese caché a partir del replay sobre características ARM.

Se conservan índices y parámetros del detector correspondientes a cada
representación, con exclusión de las frecuencias dominantes. No se sustituye
por el detector de doce variables en máscaras de seis variables.

## Implementación y trazabilidad

Un firmware conserva los 32 modelos constantes en Flash y reutiliza una
arena TFLM de 16 KiB mediante destrucción y reconstrucción del intérprete
fuera de RUN. Los buffers principales previstos son una ventana de 32768 B,
dos scratch de 16384 B y esa arena: 81920 B. El mapa debe confirmar el total
real junto con el resto del firmware; todavía no se afirma que compile o
que la arena real sea suficiente para todos los modelos.

Cada modelo se identifica mediante sus bytes y FNV calculado en selección;
sus parámetros usan una serialización explícita de 297 bytes sin padding ABI.
El catálogo canónico está enlazado por SHA256 y FNV. La placa publica SHA256
declarados, no calcula SHA256 del firmware. Se debe enlazar el ELF copiado,
la compilación y la verificación del programador antes de cerrar la evidencia.

En el protocolo UART, una consulta inválida conserva la publicación anterior;
un RUN fallido, BEGIN/DATA o SELECT la invalidan. Un SELECT rechazado por
formato o ID antes de entrar en la API de selección conserva el modelo
anterior, aunque invalida su publicación. Si la inicialización interna de
un modelo falla, la API sí descarta el modelo anterior. Las lecturas exigen
el ID de publicación y checksum correspondientes.

La Flash conjunta de este banco no es una comparación de memoria entre
candidatos. La ocupación de un despliegue final se verificará en su firmware
específico. Tampoco se hereda una medida de stack antigua como medida de
esta configuración nueva.

## Puertas y alcance de la conclusión

La puerta inicial usa representante 0 y ventana global 568: tres RUN y un
PROFILE, más 17 rechazos sin RUN válidos adicionales. La campaña completa
requiere puerta PASS con el mismo ELF y catálogo, y conservará RAW UART,
intermedios, tiempos, inicializaciones y diferencias numéricas.

Un resultado completo del cliente tendrá estado pendiente de auditoría.
Solo después de contrastar trazas, bancos, firmware y conteos podrá cerrarse
una tabla `VALIDATED_TOTAL_COST` por escenario y máscara: media y envolvente
observada [mínimo, máximo] de sus 84 RUN principales. Estas envolventes no
son intervalos de confianza ni WCET. No se modifica todavía el protocolo
sellado ni se habilita el selector.

Esta etapa no corrige por sí sola las limitaciones predictivas del detector
observadas en 0 HP. Se mantendrán explícitas en la evaluación y en el artículo.
