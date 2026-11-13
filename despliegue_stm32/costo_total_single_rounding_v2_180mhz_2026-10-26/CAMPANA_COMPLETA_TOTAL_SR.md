# Campaña completa de costo total con Single Rounding

La puerta `20261026T143621_946818Z` terminó con los dos casos aprobados:
control modelo 0/ventana 568 y regresión modelo 22 DE7/ventana 232. Contiene
seis RUN, dos PROFILE, dos BASE y 17 rechazos. El caso DE7 ahora devuelve
`[41,-100,-69]`, coincidente con PC para la misma entrada INT8. La puerta
no constituye todavía una tabla validada de costo total.

## Ejecución personal del usuario

Mantener la placa ejecutando el mismo firmware SR y dejar COM4 libre.
La campaña comienza con BEGIN y transmite de nuevo las ventanas completas;
el estado que dejaron las pruebas negativas se sustituye por el protocolo.
Si la placa sigue ejecutando ese firmware, no hace falta volver a cargarlo
ni repetir la puerta. Conservar también el log de programación/verificación
si está disponible.

Desde PowerShell, en cualquier carpeta:

```powershell
$cwruSR = "C:\Users\LENOVO\Desktop\Proyecto Investigación\despliegue_stm32\costo_total_single_rounding_v2_180mhz_2026-10-26"

& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "$cwruSR\Validacion_PC\validar_total_sr.py" --full --port COM4 `
  --firmware-elf "D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_TotalV2_SR\Prueba_2_CWRU_Extraccion.elf" `
  --build-report "$cwruSR\COMPILACION_SR.json" `
  --gate-evidence "$cwruSR\Validacion_PC\resultados\20261026T143621_946818Z" --max-seconds 1800
```

Se utiliza el Python 3.12 del usuario con TensorFlow 2.21.0, NumPy y
pyserial, como en la puerta. El cliente comprueba que la puerta, el ELF,
el catálogo, los recursos y su propio código corresponden entre sí antes
de abrir el puerto. No entrena ni reexporta modelos.

## Volumen, progreso y presupuesto

- 32 representantes P0 existentes y 42 ventanas RAW únicas.
- 896 combinaciones autorizadas de modelo y ventana.
- Tres RUN por combinación: 2688 tiempos principales, 84 por representante.
- Un PROFILE adicional por combinación: 896 diagnósticos separados.
- 42 BASE y 17 rechazos al finalizar, fuera del costo principal.

La campaña genera una captura completa nueva con el firmware SR. Los seis
RUN de la puerta sirven para comprobar el funcionamiento y no se suman
a los 84 tiempos prefijados por representante. Los tiempos del firmware
anterior, incluida su campaña parcial fallida, tampoco se importan.

Reservar 15–25 minutos, con presupuesto de 1800 s (30 minutos). Es una
previsión de planificación, no un tiempo garantizado ni una medida WCET.
La duración incluye arranque de TensorFlow, transmisión RAW por UART,
cambio de modelos, publicaciones y validaciones; no equivale a multiplicar
únicamente la latencia de inferencia. La estimación basada en la puerta SR
da una referencia de 17.21 minutos y se conserva en
`ESTIMACION_DURACION_CAMPANA_SR.json`.

El presupuesto se comprueba en los puntos de control del cliente y en las
esperas UART; cada respuesta tiene un timeout de 15 s. No interrumpe desde
fuera una importación o llamada interna de Python bloqueada.

Al iniciar se imprime una nueva ruta en `Validacion_PC/resultados/`.
Se esperan 896 líneas de avance, una por combinación, y al finalizar:

```text
FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT
```

Si aparece un error, se conserva la evidencia parcial; el ejecutor no
omite fallos ni los reintenta automáticamente. Entregar la ruta nueva y
el mensaje final para su revisión.

## Alcance del cierre posterior

Se revisarán trazas, integridad, bancos, 32 identidades de modelo y
parámetros, asignaciones de arena, igualdad INT8 y estadísticas. El selector
solo podrá usar los 84 RUN principales auditados de cada representante.
BASE, PROFILE, inicialización y UART quedan fuera de ese costo.

El estado final del cliente solo indica que hay medidas listas para
auditoría. `VALIDATED_TOTAL_COST` y la habilitación del selector requieren
el cierre posterior. Esta campaña no evalúa nuevas semillas, exactitud
externa ni independencia de motores.

Los fingerprints UART y el ELF declarado vinculan modelos y parámetros,
pero no son un readback de toda la Flash. La falta de un log del programador
en la puerta se conserva explícita como límite de identidad física.
