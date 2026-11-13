# Reajustes finales CWRU V2

Se prepara únicamente el reajuste de modelos y parámetros a partir de las
cuatro decisiones congeladas el 27 de octubre de 2026. La evaluación de la
carga reservada se realizará en una etapa posterior, una vez comprobada esta
campaña. El conjunto ya fue explorado anteriormente: sigue siendo una
evaluación retrospectiva; estas particiones no crean un motor independiente.

## Trabajos y alcance

| Familia | Trabajos únicos | Datos permitidos | Regla conservada |
|---|---:|---|---|
| DNN | 80 | Tres cargas de desarrollo, 21 MAT | Diez semillas, épocas fijas por máscara |
| RF | 40 | Tres cargas de desarrollo, F14 sin escalar | Configuración interna seleccionada, diez semillas |
| SVM | 4 | Tres cargas de desarrollo, F14 | Scaler train-only float64 exportado a float32; predicción nativa |
| Detector | 7 | Solo normales: dos cargas para ajuste y otra para calibración | K=1, P95 observado, sin reajuste tras calibrar |

Son 131 trabajos. Se comparten los entrenamientos idénticos solicitados por
varios procedimientos; no se reutilizan como finales los modelos internos.
Las 80 DNN suman 2400 épocas y 80 840 actualizaciones. El entrenamiento usa
1205/987/974/974 ventanas cuando se reserva respectivamente 0/1/2/3 HP.

La DNN mantiene arquitectura d–20–10–3, Adam, pérdida ponderada con L2,
barajado y cálculo float32 establecidos. El cambio respecto del ajuste
interno es el ya fijado para esta etapa: se ejecutan exactamente las épocas
seleccionadas, sin validación, parada temprana o restauración de un mejor
checkpoint. Se conservan los pesos de la última época. La calibración INT8
recorre todas las ventanas de entrenamiento en orden de ID global.

Se guardan diagnósticos técnicos sobre entrenamiento para comprobar
serialización, conversión y recarga. Esas cifras no estiman generalización.
Los workers reciben un NPZ de su ámbito autorizado. El almacenamiento
histórico compartido puede contener otras cargas: se verifica su hash y se
seleccionan únicamente las filas permitidas; no se afirma una traza de
accesos del sistema operativo. No se abren señales MAT ni se materializan
valores de la carga exterior del trabajo.

La limitación del escenario 3 permanece: el selector por costo devuelve
F14 como respaldo, sin cumplir el ahorro objetivo ni el margen predictivo.
Su macro-F1 interno está 7.0995 puntos por debajo del selector NF12. No se
cambia esta regla después de observar el resultado. El problema de baja
especificidad del detector tampoco se resuelve mediante estos reajustes.

## Validación técnica y conservación de intentos

`PRUEBAS_PREPARACION_FINAL.json` contiene las pruebas de contrato, ámbitos,
presupuesto, bloqueos y recuperación de trabajos completos. No entrenan
modelos CWRU ni usan valores de sus características.

`PRUEBA_SINTETICA_REAL_INTENTO2.json` recoge el ensayo técnico autorizado,
con 42 filas generadas artificialmente y procesos separados por familia.
Comprueba ajuste, guardado, recarga, conversión FP32/INT8 y repetición
determinista de dos épocas DNN. La arquitectura y la actualización Adam se
comparan con el núcleo original. La autorización de producción se sustituye
solo dentro de estas fixtures; el ejecutor real conserva sus controles.

El primer ensayo se conserva en `prueba_sintetica_real/` y
`PRUEBA_SINTETICA_REAL.json`. Su proceso RF completó el ajuste y la
comprobación de recarga, pero venció el watchdog de 30 s al terminar. El
resultado registra 29.219 s de worker, de los cuales 0.906 s fueron ajuste:
el arranque/importación dominó. Aunque el informe principal no incorporó la
familia a su contador al interrumpirse, los artefactos acreditan **un ajuste
RF sintético**. No fue un ajuste CWRU. El segundo intento usa los límites
operativos por familia y conserva los registros del primero. También se
conservan las primeras pruebas y el primer presupuesto como antecedentes;
los archivos con sufijo FINAL y el informe INTENTO2 sustentan la preparación
vigente. No se modificaron las decisiones científicas.

## Tiempo y límites

`PRESUPUESTO_Y_LIMITES_FINAL.json` contrasta los tiempos de 240 procesos DNN,
288 RF y 96 SVM internos, con 1257 fuentes de evidencia autenticadas y el
generador del presupuesto. La extrapolación considera el mayor número de
ventanas y las épocas fijas: proyección escalada aproximada **29 minutos**,
sensibilidad de máximos mixtos **63 minutos**. Su reserva histórica orientativa
es **30–75 minutos**. No es una garantía; el arranque, conversión y verificaciones
pueden cambiar. Para el detector se reserva el watchdog completo porque no
hay un perfil independiente comparable de sus procesos finales.

Los timeouts de worker son DNN 300 s, RF 90 s, SVM 60 s y detector 60 s. El
detector recibe margen adicional de arranque ante el caso sintético observado.
El presupuesto acumulado es 7200 s para preparación por trabajo, proceso y
verificación de nuevos trabajos; el preflight compartido y el replay de
trabajos ya completos quedan fuera de ese contador. El límite de lote también
es 7200 s. No se inicia otro trabajo si no queda su timeout completo. La
verificación del padre y el cierre de procesos pueden añadir tiempo sobre
esas fronteras: no es un límite duro del tiempo calendario total.

Cada worker autentica de nuevo el plan y sus referencias; los perfiles
históricos no miden exactamente ese costo adicional. La auditoría del
detector reconstruye scaler, centroide y P95, también al reanudar. Son
comprobaciones deterministas del mismo resultado, no nuevos experimentos
independientes: siete ámbitos no equivalen a solo siete llamadas internas
al núcleo numérico si se incluyen estas verificaciones.

El segundo ensayo sintético observó 50.371 s en importación y lectura de su
payload DNN. `NOTA_ARRANQUE_FRIO_Y_PRIMER_AJUSTE.json` conserva ese dato: añadir
ese tiempo íntegro a las 80 DNN, además de la proyección histórica, produce
una sensibilidad de 96.22 minutos. Ese cálculo duplica deliberadamente la
importación ya incluida; no es una predicción media ni una cota. Como margen
provisional, reservar **30–100 minutos** para todo el lote y medir primero un
reajuste real. El límite acumulado de 7200 s no cambia; aún no se entrega el
comando de los restantes hasta revisar ese primer resultado.

## Primer comando: un reajuste de comprobación

Ejecutar personalmente en PowerShell, desde cualquier directorio:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\reajustes_finales\ejecutar_reajustes.py" --run --max-new-jobs 1 --max-seconds 600
```

El primer trabajo es `outer0_F14_e45_s11`: F14, semilla 11, 45 épocas, cargas
1/2/3 HP. Su worker tiene un máximo de 300 s; el lote permite hasta 600 s
según las fronteras descritas. Si aprueba debe imprimir:

```text
outer0_F14_e45_s11: PASS
BATCH_COMPLETED: 1/131 trabajos completos.
Evidencias: ...
```

Los resultados se guardan automáticamente bajo
`C:\Users\LENOVO\Desktop\Proyecto Investigación\resultados\cwru\ampliacion_v2\reajustes_finales_v2_2026-10-27`.
La salida muestra la campaña exacta. Dentro de ella, cada trabajo conserva
modelos o parámetros, procedencia, historial cuando corresponde y manifiesto;
`pasos/` conserva estados y tiempos, `lotes/` los cierres y `solicitudes/` los
NPZ y payloads autorizados. Los logs de cada proceso quedan en la campaña.

Después de auditar ese primer resultado se podrá continuar con todos los
pendientes mediante un comando. La campaña única reconoce el prefijo PASS,
verifica sus firmas y conserva el presupuesto consumido. No iniciar comandos
simultáneos. Un FAIL, timeout, interrupción, carpeta parcial o lock activo
bloquea la continuación; conservarlo para revisión, sin borrar o forzar un
reintento. La salida final esperada es
`FINAL_REFITS_COMPLETED_NO_OUTER_EVALUATION: 131/131 trabajos completos.`

No se ha ejecutado aquí ningún reajuste CWRU, evaluación exterior, conexión
STM32 ni modificación del manuscrito.
