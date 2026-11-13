# Corrección aislada de la divergencia INT8

La campaña anterior se conserva con estado `FAILED_INCOMPLETE`. El caso
DE7/ventana 232 tiene la misma entrada INT8 en PC y STM32, pero las salidas
son `[41,-100,-69]` y `[42,-101,-69]`. La auditoría reprodujo el mecanismo:
PC utiliza redondeo simple y el kernel CMSIS-NN enlazado utiliza doble
redondeo. La diferencia nace en la primera capa y se propaga hasta Softmax.
El diagnóstico cerrado está en la carpeta histórica, sin alterar su captura.

## Cambio preparado

La nueva configuración `Release_180MHz_TotalV2_SR` conserva el pipeline,
los 32 modelos, los pesos, semillas, particiones, escaladores y umbrales.
Añade una copia exacta de la fuente original
`arm_nn_vec_mat_mult_t_per_ch_s8.c`, compilada con redondeo simple. Este objeto
de aplicación debe resolver el símbolo antes de la biblioteca precompilada.
No se recompila ni se sustituye toda la biblioteca TFLM.

Se comprobaron las rutas de los 32 modelos: tres capas INT8 por canal en
cada uno, 1056 multiplicadores FC con shifts entre −12 y −5, y Softmax con
shifts entre 21 y 24. En estos dominios no cambian las ramas extremas de
generación de parámetros. El resto de objetos de la biblioteca permanece
original. La auditoría posterior al build debe confirmar el enlace real y
el modo de redondeo emitido.

Las 16 configuraciones y sus ELF anteriores se preservan en el proyecto
original y en respaldos. Esta preparación crea una copia aislada con una
configuración 17. Todavía no instala SR en D: ni programa la placa.

## Compilar personalmente

Ejecutar desde PowerShell, en cualquier carpeta:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\stm32\preparar_total_single_rounding_v2.py" --build --timeout 1800
```

Duración estimada: unos minutos; presupuesto por intento de 30 minutos.
Compila EntM, TotalV2 anterior y SR en un workspace independiente de CubeIDE.
No entrena ni abre COM. El comando realiza un intento, sin reintentos
automáticos. Los dos controles deben conservar sus ELF e imágenes Flash.

Se guardan `CUBEIDE_SR_INTENTO_1.log`, `COMPILACION_SR_INTENTO_1.json` y los
artefactos de cada configuración. `COMPILACION_SR.json` solo se crea si los
controles automáticos pasan; su estado correcto es
`PASS_SR_BUILD_PENDING_ELF_REVIEW_AND_ARM`.

Después se revisarán ELF, instrucciones y mapa antes de instalar o cargar
la variante. Los controles deben demostrar objeto SR antes de la biblioteca,
símbolo único y ausencia del miembro antiguo extraído desde el archivo `.a`.

## Prueba física siguiente

La nueva puerta contiene dos casos: control modelo 0/ventana 568 y regresión
modelo 22/ventana 232. Exige seis RUN, dos PROFILE, dos BASE y 17 rechazos.
La comparación INT8 sigue siendo exacta: una diferencia de una unidad
continúa deteniendo la prueba. La puerta antigua de tres RUN no habilita
esta campaña SR.

La corrección aún no está comprobada físicamente. Solo después de aprobar
esa puerta se dará el comando de campaña completa. Sus tiempos deben
corresponder a un único firmware; los 66 RUN y 22 PROFILE aceptados antes
del fallo anterior quedan conservados como evidencia parcial y no se
importan a la futura tabla `VALIDATED_TOTAL_COST`.

El selector sigue deshabilitado. El manuscrito no se modificó.
