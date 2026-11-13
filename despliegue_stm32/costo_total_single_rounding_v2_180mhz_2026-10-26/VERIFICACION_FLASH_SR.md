# Lectura posterior de la Flash y correspondencia con el ELF SR

La campaña completa conserva los modelos y parámetros publicados por UART y
el ELF declarado, pero no incluye un log de verificación del programador.
Esta lectura permite contrastar la imagen presente en la placa con el ELF SR
auditado. Es una comprobación posterior; no se presentará como un log de la
programación histórica ni prueba por sí sola qué imagen existía en cada
instante de la campaña.

## Antes de ejecutar

Conservar la placa con el firmware de esa campaña, sin volver a programarla.
Terminar la sesión Debug en CubeIDE para liberar ST-LINK y cerrar cualquier
otra herramienta que utilice la sonda. Guardar los archivos abiertos sin
recompilar el proyecto. El verificador no usa COM4, no programa, no borra
Flash y no solicita reset. Usa SWD en modo HOTPLUG con la sonda
`0672FF505487525067162526` de la NUCLEO-F446RE.

La lectura la inicia personalmente el usuario desde PowerShell, en cualquier
carpeta:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\stm32\verificar_flash_total_sr_v2.py" --readback --timeout 120
```

Presupuesto solicitado: 120 s, de los que se reservan hasta cinco para cerrar
el proceso si no termina. Una lectura normal debería requerir segundos.
No hay reintentos automáticos. El comando verifica antes los hashes de CLI,
lector, ELF original/instalado y la captura completa. Si falla una guarda,
no inicia la conexión a la placa.

## Resultado y alcance

Se crea una carpeta UTC exclusiva en `lectura_flash/` dentro de esta entrega.
Conserva el dump de 512 KiB, ELF auditado, plan, stdout/stderr, log detallado,
resultado JSON y manifiesto SHA-256. Cualquier fallo conserva sus evidencias
y no sustituye las capturas anteriores.

El estado de éxito es `PASS_FLASH_READBACK_MATCHES_AUDITED_SR_ELF`. Se
comparan exactamente 280288 bytes de los dos segmentos Flash PT_LOAD del
ELF, incluida la imagen inicial de datos RAM. El resto de la Flash y los
huecos no especificados por el ELF no se fuerzan a `0xFF`. El dump completo
se conserva, pero no se afirma que todos sus bytes no utilizados estén
definidos por el ELF. Tampoco se comparan BSS o stack de RAM.

Entregar la ruta y el estado final para su revisión. La lectura no habilita
por sí misma el selector ni ejecuta nuevos ensayos predictivos.

Para comprobar únicamente los archivos y el plan, sin conexión hardware,
el mismo script acepta `--preflight` en lugar de `--readback --timeout 120`.
