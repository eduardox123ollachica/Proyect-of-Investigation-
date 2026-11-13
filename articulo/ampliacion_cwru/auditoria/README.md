# Auditoría del subconjunto CWRU

La ejecución completa y validada es [revision_2026-10-22_validada](revision_2026-10-22_validada/INFORME_AUDITORIA_CWRU.md).

## Evidencias principales

- Informe: `revision_2026-10-22_validada/INFORME_AUDITORIA_CWRU.md`.
- Inventarios por MAT y por variable, calidad/autocorrelación y distribución.
- Copias exactas, equivalencias afines y barridos completos de bloques.
- Reconstrucción independiente de todas las ventanas y características.
- Registro de 25 comprobaciones: 18 PASS, 3 WARN y 4 LIMITATION; ningún FAIL.
- Nueve pruebas sintéticas aprobadas, incluidas pruebas negativas.
- Tres imágenes diagnósticas, referencias primarias y límites de interpretación.
- 32 archivos de salida sellados por `MANIFEST_SHA256.csv`, más el propio manifiesto.
- 96 entradas con SHA256 antes/después, sin cambios.

Se ejecutó además el verificador separado, que confirmó las 32 salidas, las 96
entradas y las nueve pruebas sintéticas. La ejecución completa del análisis tardó
41.4 s en el entorno registrado; es una medición de esta ejecución, no una promesa
de duración futura ni incluye el tiempo de desarrollar y revisar los verificadores.
No hubo entrenamiento ni modificación del manuscrito, datos o resultados V1.

## Decisiones adoptadas durante la auditoría

1. Lectura de claves explícitas del manifiesto. Nunca elegir el primer vector DE
   del MAT: hay variables auxiliares de otros registros/conjuntos.
2. Conservar 201/213/238 sin alterar sus canales afínmente equivalentes. No
   eliminarlos automáticamente ni inventar un tercer eje.
3. Mantener las advertencias bibliográficas de 177/214/215 como tales. No hacer
   correcciones instrumentales sin evidencia local suficiente.
4. Aceptar únicamente redondeos decimales <=1e-12 en tiempos/diámetros de metadata.
   Claves e índices siguen siendo exactos. Se observaron 291 representaciones
   diferentes, con diferencia numérica máxima 1.78e-15 s en los tiempos.
5. Mantener íntegros el artículo, el plan maestro y el protocolo propuesto.
   El campo `full_mat_audit_pending` del JSON pertenece a la propuesta anterior;
   esta auditoría ya cubre ese análisis, pero aún no habilita entrenamiento V2.

`revision_2026-10-22` conserva un intento de desarrollo interrumpido por comparar
tiempos como texto; no es una auditoría final ni debe citarse como tal. Su nota
`NO_USAR_COMO_AUDITORIA_FINAL.md` explica la diferencia. No se movieron originales.

Para repetir o verificar, seguir el README de reproducibilidad de la carpeta
validada. El código nuevo está en `codigo/cwru/ampliacion_v2/auditoria/`.

## Pendiente antes de entrenar

Definir el papel principal/secundario de DE-solo frente a DE/FE, el tratamiento
predeclarado de calidad y la evaluación retrospectiva por carga. La igualdad de
ventanas no demuestra independencia entre motores/rodamientos ni borra el uso
anterior del test 3 HP. No presentar CWRU como solución automática de toda amenaza
a la validez ni como contribución científica suficiente por sí misma.
