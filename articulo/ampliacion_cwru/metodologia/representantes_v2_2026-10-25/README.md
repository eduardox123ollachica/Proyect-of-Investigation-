# Representantes V2: preparación y ejecución por el usuario

Esta entrega prepara los 32 ajustes internos fijados para calibrar el costo
**total** del pipeline en STM32. No ejecuta la búsqueda restante, selecciona
máscaras o evalúa resultados externos. El usuario inicia cada lote.

Los 32 ajustes son parte de los 288 internos del protocolo V2. Los modelos
del piloto de tiempos no se reutilizan: se entrenaron sin parada temprana.
La reutilización futura de estos representantes requiere firma y artefactos
idénticos. La firma incluye también la validación que determina el checkpoint.

## Alcance autorizado y conservado

| Carga externa excluida | Validación interna | Ajuste DNN | Ventanas ajuste / validación |
|---|---|---|---|
| 0 HP | 1 HP | 2 y 3 HP | 812 / 393 |
| 1 HP | 0 HP | 2 y 3 HP | 812 / 175 |
| 2 HP | 0 HP | 1 y 3 HP | 799 / 175 |
| 3 HP | 0 HP | 1 y 2 HP | 799 / 175 |

En cada fila se ajustan F14, T6, S8, NF12, NM10, NE12, DE7 y FE7, con semilla
11. El banco temporal contiene primera y última ventana de cada uno de los
catorce MAT de ajuste: 28 ventanas, tres repeticiones previstas en hardware.
Validación interna y carga externa quedan fuera de ese banco.

Los archivos NPY son almacenamiento histórico. Pueden contener filas excluidas:
el lector verifica sus hashes y los mapea en lectura, pero solo materializa las
filas autorizadas de ajuste y validación interna. No se afirma aislamiento de
accesos del sistema operativo ni se restaura un test históricamente inédito.

## Primer comando: un representante

En PowerShell, desde cualquier carpeta:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\representantes\ejecutar_representantes.py" --execute --limit 1
```

El primer trabajo es `cost_outer0_inner1_F14_seed11`: ajusta con 2/3 HP y
valida con 1 HP; no calcula predicciones de 0 HP. Se recomienda auditar este
primer resultado antes de continuar. La consola debe finalizar con
`BATCH_COMPLETED: 1/32 representantes completos.` y la ruta de la campaña.
Este cierre de lote no significa que hayan terminado los 32.

Estimación orientativa del primero: alrededor de 30–120 segundos, según el
arranque de TensorFlow, más verificaciones e I/O. Límite del worker: 300 s.
No es una garantía. El protocolo permite hasta 50 épocas; ES puede terminar antes.

Salida:

`resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/<fecha_UTC>/`

Se guardan logs por proceso en `lotes/<fecha_UTC>/`. Dentro de la carpeta de
cada modelo, `progreso.jsonl` registra cada época; `historial.jsonl` conserva
pérdidas, actualizaciones, mejor época y paciencia. No debe lanzarse otra campaña
para continuar la existente: el marcador de campaña única evita duplicaciones.

## Continuación y límites

Después de auditar el primer lote se puede usar el mismo ejecutor con
`--execute --resume` seguido de **la ruta absoluta de la campaña impresa por el
primer comando**, y `--limit 31`. Ese comando se concretará con la ruta real.
Se validan los sellos y se omiten exclusivamente ajustes completos con la firma
esperada. No hay reintentos automáticos, `force` ni sobrescritura.

Un fallo, timeout, interrupción o artefacto parcial bloquea la reanudación.
Se conservan los archivos para preparar una recuperación técnica separada.
Ctrl+C durante la espera termina al hijo activo. Si el sistema se cierra de
forma abrupta puede quedar un lock: auditar antes de retirarlo.

La campaña tiene topes de 32 ajustes y 64 conversiones, 300 s por worker y
7200 s acumulados de procesos y comprobaciones asociadas del padre. Son límites
operativos nuevos, no cotas estadísticas. No incluyen pausas del usuario,
compilación/flasheo, UART ni adquisición. Los 32 topes individuales sumarían
9600 s: el límite global de 7200 s puede detener la campaña antes de agotarlos.

Los pilotos sugieren del orden de 15–30 minutos para los 32 con importaciones
habituales; el escenario de arranques fríos puede acercarse a una hora o más.
Esta es una orientación derivada de pilotos y tamaños parecidos, sin medición
del ejecutor definitivo en CWRU. La primera ejecución permitirá afinarla.
La referencia previa de 3 h 26 min corresponde al plan CPU completo, no a estos 32.

## Qué produce cada ajuste

- Procedencia, roles, IDs globales, MAT, hashes y versiones.
- Scaler DNN train-only en float64 y parámetros exportados float32.
- Historial y checkpoints de mejora; mejor peso restaurado y guardado **antes**
  de convertir. ES: CE de validación sin pesos más L2, paciencia 10, mejora
  estricta mayor de 1e-12. Los pesos de clase se calculan solo con ajuste.
- Modelos FP32 y full INT8, congelación comparada a lote 1, parámetros de
  cuantización, saturaciones de ajuste/validación/banco y cambios FP32–INT8.
- Predicciones y métricas por clase, matriz de confusión y macro-F1 INT8 de
  validación interna; no eligen por sí solas ninguna máscara.
- Detector normal-only independiente, P95 y scores. Ajuste N de la menor carga
  de entrenamiento, calibración N de la mayor, sin refit posterior.
- Banco PC de 28 referencias de características, entradas/salidas INT8 y scores
  para preparar la medición física posterior. No contiene latencia STM32.
- Resultado y manifiesto SHA-256 por ajuste, estados y manifiesto por lote.

El detector se genera para cada representante para dejar un paquete autónomo.
F14 y NF12 comparten sus doce entradas de detector; estos cálculos duplicados
no son evidencias independientes ni repeticiones multisemilla. Más adelante
se podrá reutilizar su artefacto solo bajo la firma de scope numérico idéntica.

El auditor del lote cruza etiquetas e IDs con el CSV sellado, reproduce métricas
desde salidas guardadas, comprueba el P95, las métricas del detector y los pesos
restaurados frente a la mejor época. No vuelve a entrenar ni ejecuta inferencia.
No equivale a una reproducción independiente en otra máquina.

## Comprobaciones antes de entregar

Se verificaron los treinta recursos del protocolo, los bancos de los 32 y
lectura de los cuatro pliegues. Las 44 pruebas de preparación cubren roles,
fugas, índices, cálculos, firma, timeouts y preservación, con cero fallos u
omisiones en el Python del proyecto. No entrenan modelos CWRU.

La prueba TensorFlow usa doce filas sintetizadas y seis de validación, tres
actualizaciones y dos exportaciones. Verifica restauración, checkpoint previo,
congelación a igual lote y full INT8. La paciencia se prueba con curvas
sintéticas puras; tres épocas no prueban una parada temprana real de TensorFlow.
Se conserva un primer ensayo sintético con oneDNN activado; el ensayo vigente
restaura las variables deterministas del piloto. Ninguno usa datos CWRU.

Los avisos de deprecación del conversor/interprete de TensorFlow 2.21 se
conservan. Se mantiene la versión comprobada; no se migra de biblioteca durante
estos ajustes. La comprobación de tipos y operadores impide aceptar fallback
flotante aunque el conversor imprima avisos.

## Lo que falta después

Cerrar y auditar los 32, preparar el firmware de cada representante y medir
directamente el pipeline completo. Un auditor posterior deberá vincular modelos,
scalers, detector, firmware y trazas con cobertura de 28×3 RUN por candidato.
Solo esa evidencia podrá producir `VALIDATED_TOTAL_COST`. La tabla actual de
extractores y los tiempos PC no sirven para rellenar los términos faltantes.

La búsqueda de los 256 internos restantes, los comparadores predictivos,
selección bloqueada por pliegue, evaluación externa y actualización del artículo
siguen pendientes. No se modificó el manuscrito o firmware desplegado.
