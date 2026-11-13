# Preparación de la búsqueda interna DNN V2

Esta etapa completa la matriz DNN interna del protocolo ya fijado: cuatro
escenarios exteriores, tres cargas de validación interna por escenario,
ocho máscaras y semillas 11, 22 y 33. Son 288 celdas. La preparación no
entrena ni ejecuta inferencias TFLite, y no evalúa predictivamente la carga
exterior de ningún trabajo.

## Reutilización y alcance

Se reutilizan 48 ajustes completos y auditados: los 32 representantes de
costo y 16 P0 adicionales del suplemento. Los dos P0 del suplemento que ya
eran referencias importadas no se vuelven a contar. P1 y MOM4 no se
incorporan a esta matriz. Quedan **240 ajustes DNN nuevos**.

Cada referencia conserva su firma original, manifiesto, modelos, checkpoint
y procedencia. La equivalencia entre los dos ejecutores antiguos y el nuevo
se comprueba mediante roles, IDs, máscara, semilla, datos, versiones y el
mismo núcleo de entrenamiento/exportación. Una firma del wrapper nuevo no
se presenta como textualmente idéntica a la firma original.

La DNN conserva arquitectura d–20–10–3, Adam 0.005, lotes de 32, L2 0.001 y
pesos de clases calculados exclusivamente con entrenamiento. Se mantiene
EarlyStopping con paciencia 10, mejora mínima 1e-12 y restauración del mejor
checkpoint, hasta un máximo de 50 épocas. Las 50 épocas fijas correspondían
solo al piloto temporal. La cuantización INT8 usa todas las filas de
entrenamiento del trabajo, en el orden definido, sin calibrar con validación.

El worker nuevo utiliza las funciones originales de configuración de
TensorFlow, entrenamiento, exportación FP32/INT8 e inferencia de validación.
El escalador se ajusta con las filas de entrenamiento y se materializan sus
operaciones float32 según el contrato previo.

Los cachés con nombres históricos train/validation/test son almacenamiento:
los roles de cada trabajo se asignan por carga e IDs del índice sellado.
Los ficheros se comprueban y mapean en lectura; solo se seleccionan las
filas autorizadas de entrenamiento y validación interna. La comprobación
de hashes o el mapeo del fichero no se presenta como una traza de todos
los accesos del sistema operativo. No se usan señales RAW ni MAT nuevos.

El ejecutor de esta entrega es **solo DNN interna**. No vuelve a ajustar
detectores por semilla, ni entrena RF/SVM, ni hace selección de máscara,
ajustes externos, programación STM32 o cambios al manuscrito. Las filas del
detector faltantes y la búsqueda clásica siguen como bloques separados;
su inventario se documenta para no confundir una matriz DNN terminada con
una selección completa.

El detector tiene 36 de 96 filas lógicas disponibles: faltan 60 filas.
Al compartir parámetros entre roles equivalentes y las máscaras F14/NF12,
son 23 de 42 ámbitos de parámetros distintos disponibles y 19 pendientes.
Estas cantidades no son repeticiones independientes y no se multiplican
por las tres semillas de la DNN. El piloto temporal RF/SVM ya realizado
también se conserva.

## Primera ejecución del usuario

Comprobar que la preparación está cerrada y que el preflight del entorno
Python del usuario aprobó. Desde PowerShell, en cualquier carpeta:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\busqueda_interna\ejecutar_busqueda_interna.py" --run --max-new-jobs 1 --max-seconds 600
```

Este comando inicia **un solo entrenamiento nuevo**, además de sus dos
conversiones y evaluación interna. Los 48 ajustes importados no se repiten.
El trabajo usa CPU y un hilo, con el entorno determinista fijado por el
ejecutor. No necesita la placa ni COM4.

Se imprime la ruta de campaña y el identificador del trabajo. La evidencia
se guarda en `resultados/cwru/ampliacion_v2/busqueda_interna_v2_2026-10-26/`.
El resumen del primer lote debe cerrar sin fallos y dejar 49/288 celdas
disponibles. Entregar la ruta para revisar ese primer ajuste antes de
continuar con un lote mayor. Una línea INICIO no implica que haya terminado.

El presupuesto solicitado al primer lote es de 600 s, con límite de 300 s
para el proceso de entrenamiento. Los trabajos recientes suelen requerir
decenas de segundos, pero otra semilla, la importación de TensorFlow o la
carga del equipo pueden aumentar el tiempo. El informe de presupuesto
separa la previsión del tiempo de los límites de ejecución.

El contador acumulado de 7200 s incluye el lanzamiento del proceso y su
comprobación posterior por ajuste. Es algo más conservador que contar solo
el subproceso. No incluye toda la preparación y verificación entre lotes ni
el tiempo que el usuario espere entre comandos; no es un límite de tiempo
calendario global.

## Continuación y recuperación

Tras revisar el primer ajuste, se entregará el comando con `--campaign`
apuntando a la campaña existente, para lotes de hasta 16 ajustes nuevos y
1800 s por invocación. La previsión de los 240 ajustes nuevos es del orden
de una a dos horas de trabajo total; los lotes permiten revisar progreso
y detenerse entre invocaciones. El límite acumulado y la base de estimación
se fijan en el plan y el presupuesto de esta carpeta.

Los procesos son secuenciales y el ejecutor guarda resultados y manifiestos
incrementalmente. Solo reutiliza trabajos completos, PASS y con firmas
coincidentes. Un fallo, timeout, interrupción o archivo alterado detiene la
cola; no se omite una celda de la matriz ni se reintenta automáticamente.
La incidencia se conserva para revisar una recuperación técnica separada.
Una parada prevista en el límite de presupuesto, antes de iniciar otro
trabajo, se distingue de un fallo de entrenamiento.

No borrar carpetas o locks para forzar una repetición. La reanudación usa
la campaña existente y los hashes previstos; las ejecuciones ya completas
no se sobrescriben. La preparación del agente no inicia trabajos reales.

## Resultado científico posterior

Completar esta matriz permite evaluar la variación de las tres semillas
internas y alimentar la selección cuando también estén completas las
restricciones del detector. Los costos STM32 ya validados se conservan como
insumo por escenario y máscara; no se recalculan con predicciones externas.
Las semillas representan variación de entrenamiento, no adquisiciones
independientes de distintos motores o rodamientos. La evaluación continúa
siendo retrospectiva y la fase exterior sigue pendiente.
