# Verificación física de la corrección Single Rounding

La ejecución `20261026T143621_946818Z`, iniciada por el usuario el
26 de octubre de 2026, terminó en
`SR_GATE_PASS_NOT_TOTAL_COST_VALIDATED`. Se ensayaron modelo 0 F14/ventana
568 y modelo 22 DE7/ventana 232, con tres RUN principales y un PROFILE por
caso. También se obtuvieron dos BASE y 17 rechazos correctos del protocolo.
El cliente informó 27.922 s para el lote, incluyendo sus operaciones PC.

La auditoría reprodujo íntegramente 834 filas UART, 417 transacciones y
30 eventos. Reconstruyó ambas ventanas, con 171 tramas y 8192 valores
float32 cada una, idénticos al banco autorizado y a sus filas fuente.
El contador final fue ocho, sin ejecuciones positivas adicionales; las
pruebas negativas dejaron invalidada la publicación según el protocolo.

## Resultado de la regresión

DE7/232 conserva la entrada INT8
`[-45,-15,-38,-36,-69,-39,-42]`. Con el firmware anterior devolvía
`[42,-101,-69]`, frente a `[41,-100,-69]` de PC. En esta ejecución, los tres
RUN y el PROFILE devuelven `[41,-100,-69]`, exactamente iguales a PC. El
caso de control conserva `[127,-128,-128]`. Se mantiene la comparación
exacta, sin añadir tolerancias ni cambiar pesos, escaladores o umbrales.

Esto aporta una comprobación física de la corrección en el caso concreto
que originó el diagnóstico. Junto con el replay entero y las instrucciones
del ELF auditado, respalda la explicación por diferencia de redondeo.
Dos casos no establecen equivalencia universal de los 32 representantes;
la campaña completa sigue siendo necesaria.

## Tiempo y memoria observados

| Caso | RUN principales | Media (ms) | Mínimo (ms) | Máximo (ms) |
|---|---:|---:|---:|---:|
| F14 / 568 | 3 | 84.348869 | 84.333894 | 84.357239 |
| DE7 / 232 | 3 | 42.126904 | 42.124606 | 42.128761 |

Los tiempos proceden de CYCCNT con HCLK nominal HAL de 180 MHz e incluyen
extracción, escalado/cuantización, copias de entrada/salida, Invoke, argmax
y detector. UART, inicialización, BASE y PROFILE no entran en estas medias.
No se descuenta la calibración de lecturas ni se afirma WCET. Tres medidas
por caso no sustituyen los 84 tiempos prefijados por representante.

La asignación informada de arena es 1188 B para ambos representantes,
frente a 16384 B reservados. Esta puerta no verifica todavía la asignación
de los otros 30 modelos ni mide de nuevo el máximo de stack.

## Relación entre referencias y evidencia

Se distingue la comparación con el caché PC original de la comparación
para una misma entrada realmente procesada por ARM. La primera admite
las tolerancias numéricas ya fijadas para las características; la segunda
exige intermedios float32 reproducibles y salida INT8 exacta. F14 presentó
una ULP de diferencia en energía DE frente al caché. La BASE de la ventana
232 presentó dos ULP en energía DE y FE y una ULP en amplitud FE; el modelo
DE7 utiliza las características DE. Estas diferencias no alteraron las
entradas INT8, las clases ni las decisiones del detector. No se describen
como identidad binaria de todo el preprocesamiento PC–ARM.

La captura vincula el ELF SR
`c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7`,
el cliente y el catálogo mediante hashes. No se adjuntó log del programador
a esta ejecución; por tanto, no se afirma una verificación independiente
de toda la imagen Flash cargada. Los fingerprints publicados sí se
contrastan con los modelos y parámetros autorizados.

Las auditorías de traza y numérica se guardan separadas, junto con su
cierre, en `despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/`.
La captura original y sus sellos se conservan sin modificación. El
manuscrito no se modifica en esta etapa.

## Paso siguiente

Ejecutar personalmente la campaña SR completa con 42 ventanas únicas,
896 combinaciones, 2688 RUN principales y 896 PROFILE. Se conservará una
captura completa de un mismo firmware; los tiempos del ensayo antiguo
fallido no se incorporarán a la futura tabla. El selector permanece
deshabilitado hasta auditar esa campaña.
