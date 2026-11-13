# Auditoría de la campaña completa de costo total SR

La campaña `20261026T144540_396879Z` completó 2688 RUN, 896 PROFILE, 42 BASE y 17 rechazos. Duró 981.047 s (16 min 21 s). Las auditorías de protocolo y numérica aprobaron sin repetir el ensayo.

Se reconstruyeron 45210 filas UART y 22605 transacciones. Las 42 ventanas transmitidas contienen 344064 valores float32, exactos al banco y a las filas fuente. Cada uno de los 32 representantes tiene 28 ventanas de sus dos cargas de entrenamiento y 84 RUN principales. Se excluyen las cargas externa y de validación interna de su banco.

En las 3584 publicaciones, el replay desde las características ARM reproduce escalado, cuantización y detector. Las entradas INT8, salidas, clases y decisiones coinciden con las referencias PC selladas. Las características float32 originales pueden diferir dentro de las tolerancias fijadas; no se afirma identidad binaria de todo el preprocesamiento. Se calcularon además 2688 capas FC enteras offline (29568 neuronas); no son activaciones internas capturadas en la placa.

## Tiempos principales por escenario

Medias en ms, con 84 observaciones por celda. El escenario identifica la carga externa excluida, no una medición sobre esa carga. El JSON conserva mínimos, máximos, mediana, P95, desviación y bancos exactos por representante.

| Máscara | Externo 0 HP | Externo 1 HP | Externo 2 HP | Externo 3 HP | Condición temporal 20 % en los cuatro |
|---|---:|---:|---:|---:|:---:|
| F14 | 84.130482 | 84.130617 | 84.133503 | 84.138796 | No |
| T6 | 39.570316 | 39.571524 | 39.573792 | 39.579559 | Sí |
| S8 | 54.292290 | 54.292053 | 54.290943 | 54.297385 | Sí |
| NF12 | 84.525355 | 84.528803 | 84.529473 | 84.536163 | No |
| NM10 | 61.722661 | 61.724993 | 61.720464 | 61.725850 | Sí |
| NE12 | 55.024936 | 55.023218 | 55.030185 | 55.034878 | Sí |
| DE7 | 42.033264 | 42.033312 | 42.034473 | 42.038419 | Sí |
| FE7 | 42.019000 | 42.019379 | 42.019487 | 42.022753 | Sí |

La regla congelada utiliza L=mínimo y U=máximo observado, con `U_candidato <= 0.8 * L_F14`. La columna temporal no aplica las guardias de F1 y detector y no selecciona un modelo. P95 y media son descriptores, no sustituyen esa regla. NF12 no redujo el tiempo respecto de F14: quitar características no asegura un ahorro proporcional.

## Memoria y alcance

Los 32 modelos informaron 1188 B usados de una arena reservada de 16384 B. La Flash de 280276 B y RAM con reservas de 88460 B corresponden al banco de 32 modelos. Esta campaña no vuelve a medir el máximo de stack ni la ocupación de un despliegue final.

## Evidencia pendiente antes de habilitar los costos

El diseño previo `calibracion_costo_total_v2_2026-10-25/ALCANCE_Y_DISENO.md` exige vincular el ELF, la compilación y la verificación del programador antes del cierre. La captura no adjunta ese registro y no se localizó un log vinculable. El ELF declarado y los fingerprints de modelos y parámetros sí coinciden, pero no verifican todos los bytes cargables del programa en Flash.

Se prepara una lectura posterior por SWD para contrastar los segmentos Flash del ELF auditado sin reprogramar. Será evidencia posterior de correspondencia y no un log histórico contemporáneo a la campaña. Hasta revisarla, la tabla tiene estado `AUDITED_TOTAL_COST_PENDING_FLASH_IDENTITY`, y sus filas no son `VALIDATED_TOTAL_COST`. El selector permanece deshabilitado.

## Continuidad

No se modifica el protocolo sellado ni se rellenan sus campos históricos. Los 32 ajustes de calibración forman parte de los 288 internos previstos y podrán reutilizarse si coinciden las firmas. La selección también requiere completar las tres semillas y las tres validaciones internas por escenario; no procede elegir una máscara solo por estos tiempos. Antes del resto de entrenamientos se preparará el ejecutor reanudable y se congelarán sus límites, entregando el comando al usuario. El piloto RF/SVM ya está cerrado y no se repite.

Las capturas y sellos anteriores se conservan. No se ejecutaron nuevos entrenamientos, inferencias TFLite ni mediciones físicas durante esta auditoría; el manuscrito no se modificó.
