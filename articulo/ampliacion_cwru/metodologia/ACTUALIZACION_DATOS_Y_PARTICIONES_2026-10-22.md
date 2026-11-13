# Actualización operativa: datos y particiones CWRU V2

Estado: contrato de datos preparado; **no protocolo experimental completo congelado**.

Esta actualización continúa el plan maestro tras completar la auditoría. Se guarda
como versión nueva porque el plan y JSON originales están incluidos entre las 96
entradas selladas de la auditoría. No se regeneran esos sellos ni se reescribe la
historia. Para estos puntos prevalece `protocolo_v2_particiones_2026-10-22.json`.

## Decisiones incorporadas

1. El objetivo es comparar rendimiento y costo en cargas excluidas del banco.
   Correlación cero no es requisito de validez; independencia entre motores o
   rodamientos nuevos tampoco se presenta como resultado demostrado.
2. Mantener 12 kHz, N=4096, salto de 2048 y 50 %: era la decisión vigente. La conversación
   posterior aclaró que no era necesario eliminarlo para legitimar el estudio;
   no autorizó sustituirlo por 0 %. El análisis sin solapamiento queda como sensibilidad
   propuesta. No se elige el porcentaje según la mejor exactitud externa.
3. Conservar DE/FE como referencia F14, explicitando que son posiciones y que
   201/213/238 tienen canales afínmente equivalentes. DE7/FE7 ya están propuestos
   para comparación, sin borrar originales ni considerarlos adquisiciones nuevas.
4. Mantener los 28 MAT sin reparaciones/exclusiones automáticas. Las advertencias
   sobre 177/214/215 quedan documentadas. No se “limpia” según errores del test.
5. Declarar retrospectividad: los registros ya fueron explorados; cambiar el
   reparto no produce un test virgen ni independencia física.

## Particiones que se preparan ahora

Cuatro evaluaciones externas, cada una reservando una carga completa. Dentro de
las tres cargas disponibles se rota una para validación y dos para entrenamiento.
El registro MAT es el grupo operativo; ambos canales y todas sus ventanas reciben
el mismo papel. Cada ventana conserva su archivo y fila de origen.

Los manifiestos distinguen `development` de `outer_test`; las particiones internas
usan `inner_train`, `inner_validation` y `outer_excluded`. En el reajuste externo,
las tres cargas de desarrollo son entrenamiento, con reglas determinadas sólo
en desarrollo. No se permite usar el test externo para early stopping, scaler,
pesos de clase, selección de máscara, calibración INT8 o umbral del detector.

Se reutilizan las características **sin escalar** ya auditadas. Los antiguos
nombres train/validation/test de los NPY sólo identifican su ubicación histórica;
los roles V2 los define el manifiesto del pliegue. Un modelo o scaler V1 no se
reutiliza para estimar un pliegue que contiene datos ya vistos por ese artefacto.

## Qué comprueban las pruebas y qué no

Comprobarán cobertura, papeles, disyunción de registros/ventanas dentro de cada
pliegue, exclusión externa de los ajustes internos, claves correctas y hashes.
Incluirán casos negativos de contaminación de carga, duplicación de registros y
selección de filas externas para ajuste. No certifican independencia física ni
que un futuro entrenamiento respete el contrato si no integra estos controles.

Las particiones se materializan como CSV/índices, no copiando señales ni moviendo
archivos. Se conservan los 28 originales, las 1380 ventanas y los resultados V1.
La validación no genera nuevas métricas predictivas.

## Siguiente condición de avance

Tras validar las particiones, confirmar viabilidad y perfil de costo CWRU en
STM32, y delimitar el procedimiento de selección frente a trabajos previos.
Después: piloto acotado, presupuesto y congelamiento de las reglas restantes.
No iniciar las 408 DNN máximas de la propuesta ni evaluar los pliegues externos
antes de cerrar esas reglas. El piloto deberá limitarse a desarrollo y registrar
tiempo, recursos y condiciones; no se anuncia duración antes de medirla.

## Evidencia de partida

- Auditoría: `articulo/ampliacion_cwru/auditoria/revision_2026-10-22_validada/`.
- Solapamiento: `articulo/ampliacion_cwru/analisis_solapamiento/revision_2026-10-22/`.
- Propuesta histórica: `articulo/ampliacion_cwru/protocolo_v2_propuesto.json`.
- Plan: `articulo/ampliacion_cwru/PLAN_MAESTRO_AMPLIACION_CWRU.md`.

La skill `revision-articulo-tinyml` orienta la distinción entre evidencia medida,
límites de generalización y decisiones aún propuestas. No se modificó el manuscrito.
