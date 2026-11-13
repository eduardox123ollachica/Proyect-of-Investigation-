# Herramientas del experimento multisemilla E1--E4

Esta carpeta contiene solamente el orquestador y el consolidador. Los resultados
se conservan en `resultados/arciniegas/multisemilla/ejecuciones`, los registros
en `resultados/arciniegas/multisemilla/logs` y los productos consolidados usados
por el manuscrito en `articulo/resultados_derivados/multisemilla`.

El orquestador usa las semillas 11, 22, 33, 42, 44, 55, 66, 77, 88 y 99. Sin
`--execute` funciona en modo de inspección y solo muestra los comandos previstos:

```powershell
python codigo/arciniegas/experimentos_multisemilla/ejecutar_multisemilla.py
```

Para ejecutar se debe indicar un Python que incluya TensorFlow, NumPy, pandas,
scikit-learn y matplotlib:

```powershell
python codigo/arciniegas/experimentos_multisemilla/ejecutar_multisemilla.py --execute --python RUTA_AL_PYTHON
```

Cada ejecución se guarda bajo `ejecuciones/e1`, `ejecuciones/e2`,
`ejecuciones/e3` o `ejecuciones/e4`. El proceso se detiene antes de sobrescribir
una salida existente. El entorno histórico de la fase 4 se documenta en
`resultados/arciniegas/multisemilla/entorno_ejecucion_fase4.txt`.
