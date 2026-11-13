# Puerta física de regresión Single Rounding

Esta prueba comprueba la corrección del redondeo con el caso que detuvo
la campaña anterior y con un caso de control. No entrena modelos ni
programa la placa. El comando Python abre COM4 únicamente cuando lo
ejecuta el usuario. La campaña completa y el selector siguen pendientes.

## Preparación en CubeIDE

1. Comprobar que existe `APLICACION_SR_EN_D.json` con estado
   `PASS_SR_INSTALLED_PENDING_ARM` en esta carpeta. Ese registro confirma
   la instalación de los archivos auditados, no su carga en la placa.
2. Recargar el proyecto `D:\STM32IDE\Prueba_2_CWRU_Extraccion` para que
   CubeIDE lea la nueva configuración `Release_180MHz_TotalV2_SR`.
3. En Debug Configurations, seleccionar
   `Prueba_2_CWRU_Extraccion_TotalV2_SR_180MHz_SinRecompilar`. La entrada
   descarga el ELF auditado de `Release_180MHz_TotalV2_SR`; tiene
   desactivada la recompilación previa. No usar la entrada TotalV2 antigua.
4. Conservar el texto de la descarga y de la verificación del programador.
   Si el depurador se detiene en `main`, pulsar Resume (F8).
5. Dejar COM4 disponible para Python, cerrando cualquier monitor serie.

ELF esperado, SHA-256:
`c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7`.
El cliente contrasta el archivo ELF y los identificadores de modelos y
parámetros publicados por UART. Esto, por sí solo, no constituye un
readback criptográfico de toda la Flash de la placa.

## Comando del usuario

Desde PowerShell, en cualquier carpeta:

```powershell
$cwruSR = "C:\Users\LENOVO\Desktop\Proyecto Investigación\despliegue_stm32\costo_total_single_rounding_v2_180mhz_2026-10-26"

& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "$cwruSR\Validacion_PC\validar_total_sr.py" --gate --port COM4 `
  --firmware-elf "D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_TotalV2_SR\Prueba_2_CWRU_Extraccion.elf" `
  --build-report "$cwruSR\COMPILACION_SR.json" --max-seconds 600
```

Se utiliza el Python del usuario con TensorFlow, NumPy y pyserial.
Duración estimada: unos minutos, incluyendo el arranque de TensorFlow;
presupuesto solicitado al cliente: 600 s. La importación de TensorFlow y
una operación externa bloqueada pueden añadir tiempo fuera de los puntos
de comprobación del presupuesto; no se promete un corte externo exacto
al segundo 600.

El cliente imprime primero la ruta de evidencias y después el avance por
caso. Los avisos habituales de oneDNN o de deprecación del intérprete no
son el criterio de aprobación.

## Aceptación y continuación

Los casos prefijados son modelo 0/ventana 568 y modelo 22 DE7/ventana 232.
Se requieren seis RUN principales, dos PROFILE separados, dos BASE y
17 rechazos correctos. Se conserva la exigencia de igualdad INT8 exacta
para la misma entrada. Para DE7/232, la referencia del diagnóstico anterior
fue `[41,-100,-69]`; el cliente obtiene nuevamente la referencia del modelo
con la entrada realmente publicada por ARM, en lugar de aceptar un valor
constante escrito a mano.

El estado de éxito del cliente debe ser
`SR_GATE_PASS_NOT_TOTAL_COST_VALIDATED`. Una sola línea PASS no cierra
la prueba. Las evidencias se guardan en
`Validacion_PC/resultados/<marca_UTC>/` e incluyen resultado, trazas y
manifiesto SHA-256. Entregar esa ruta y el log de programación para la
revisión posterior, sin ejecutar todavía `--full`.

Si falla, se conserva la ejecución parcial y se revisa la divergencia antes
de repetir. La puerta antigua no sirve para esta variante. Los tiempos de
la campaña anterior no se incorporarán a la tabla de costos SR.

Esta puerta comprueba dos casos y no demuestra equivalencia sobre todos
los bancos. Después de auditarla se habilitará la campaña completa de
32 representantes, sin repetir entrenamientos ni cambiar las particiones.
