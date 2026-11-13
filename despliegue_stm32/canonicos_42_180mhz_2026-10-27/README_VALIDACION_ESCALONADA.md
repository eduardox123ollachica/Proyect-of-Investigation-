# Validación física de los ocho modelos canónicos

Fecha de preparación: 27 de octubre de 2026 (America/Lima).

## Decisión y alcance

El usuario eligió una validación escalonada: extracción y procesamiento RAW
en 56 ventanas, y procesamiento posterior sobre las características PC de
todas las ventanas reservadas. La semilla 42, los pesos, scalers, umbrales,
máscaras y épocas proceden de los reajustes y de la evaluación exterior ya
cerrados. Esta etapa no selecciona nuevos modelos.

**Estado: preparación completa; compilación, auditoría del ELF y validación física
pendientes.** Los resultados exteriores son resultados PC. Los tiempos de los
representantes internos anteriores no sustituyen las mediciones canónicas.

## Modelos y procedimientos representados

| ID UART | Carga exterior (HP) | Máscara | Épocas | Procedimientos que representa |
|---:|---:|---|---:|---|
| 0 | 0 | F14 | 45 | Base y selector por rendimiento |
| 1 | 0 | T6 | 47 | Selector por costo |
| 2 | 1 | F14 | 14 | Base |
| 3 | 1 | FE7 | 27 | Selectores por rendimiento y costo |
| 4 | 2 | F14 | 19 | Base |
| 5 | 2 | T6 | 32 | Selectores por rendimiento y costo |
| 6 | 3 | F14 | 9 | Base y respaldo del selector por costo |
| 7 | 3 | NF12 | 47 | Selector por rendimiento |

La coincidencia de varios procedimientos en un modelo se conserva como
alias, no como un ajuste o una medición independientes. El respaldo F14 en
3 HP no cumplió la meta interna de ahorro. Esta validación no cambia esa
decisión ni permite convertirla en un ahorro conseguido.

## Dos bancos complementarios

**RAW:** primera y última ventana disponibles de cada uno de los 28 MAT:
56 ventanas únicas. Cada modelo se aplica únicamente a las 42 ventanas de
sus tres cargas de desarrollo. Son 336 pares modelo–ventana. Se prevén tres
RUN principales por par (1008), un PROFILE separado por par (336), y una
extracción BASE por ventana única (56). La selección es mecánica, no depende
de la exactitud de los resultados. Las dos señales son posiciones DE y FE,
no dos ejes de un mismo sensor.

El RUN principal medirá el pipeline completo desde el RAW ya recibido en
memoria. PROFILE desglosará los costos en una ejecución distinta; BASE
servirá como referencia del extractor de 14 características. La recepción
UART y la adquisición física no forman parte de ese tiempo de cómputo.

**POST:** las 1380 ventanas exteriores reservadas, con sus 14 características
PC guardadas, se enviarán a los dos modelos canónicos de su carga exterior:
2760 pares modelo–ventana. Se verificarán escalado, cuantización, DNN y
detector. Las referencias son las salidas ya guardadas de la evaluación
exterior; no se vuelve a entrenar o evaluar un clasificador PC para preparar
el banco. La nueva ruta `CWRUTOTAL_POST` declara cero extracciones. Su tiempo
solo corresponde a las etapas posteriores y nunca se presenta como tiempo
del pipeline RAW.

La cobertura RAW es parcial. Un PASS del banco POST no demostraría por sí
solo equivalencia de extracción en las 1380 ventanas reservadas. Las
diferencias al comparar características PC con características ARM se
deberán informar separadas de la equivalencia para una misma entrada.

## Criterios que debe comprobar el cliente físico

- Identidad del firmware, catálogo y parámetros; 180 MHz, DWT activo,
  redondeo al más próximo con empate al par y FTZ desactivado.
- Extracción RAW contra PC: tolerancias heredadas `rtol=0.001` y
  `atol=0.00001`; frecuencias dominantes exactas.
- Extractor de máscara contra BASE: igualdad de bits de las características
  activas; posiciones inactivas con cero positivo.
- Procesamiento posterior: comparación exacta para la misma entrada,
  incluidas entradas y salidas INT8, clase y decisión del detector.
- Repeticiones y tiempos RAW registrados por separado del perfilado, la
  inicialización del intérprete, la transmisión y las pruebas POST.

Los gates físicos se ejecutarán antes del lote completo. El cliente de esos
gates se cerrará después de revisar el ELF compilado; no debe reutilizarse
sin adaptación el validador de los 32 representantes anteriores.
Ese cliente y la auditoría posterior se guardarán fuera de la carpeta de
generadores `codigo/cwru/ampliacion_v2/canonicos_stm32`: su inventario Python
queda sellado con la preparación y no debe cambiar durante el build.

## Aislamiento y recursos

- `recursos/`: ocho modelos INT8 copiados sin conversión, catálogo y
  parámetros autenticados; manifiesto independiente.
- `bancos/`: bancos RAW/POST y sus referencias, con otro manifiesto.
- `copia_aislada/Prueba_2_CWRU_Extraccion/`: copia del proyecto fuente.
- `PREPARACION_PROYECTO.json` y su manifiesto: inventarios de origen y copia,
  hashes y cambios concretos de configuración.
- `DOMINIO_MODELOS_CANONICOS.json`: auditoría estática de los ocho modelos y
  sus 264 canales FC en el dominio de la corrección single-rounding.

La configuración nueva se llama `Release_180MHz_Canon42_SR`; usa
`CWRU_Canon8`, un catálogo de ocho modelos y la ruta POST. Conserva las otras
17 configuraciones y el kernel single-rounding ya validado. La preparación
y el comando de compilación no escriben en el proyecto original de D:, no
abren COM y no programan la placa. La revisión posterior debe verificar el
ELF, sus modelos y parámetros, la biblioteca enlazada y la memoria antes de
autorizar su uso como firmware de validación.

## Próximo comando: compilación por el usuario

Desde PowerShell, sin necesidad de cambiar de carpeta:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\canonicos_stm32\preparar_proyecto.py" --build --timeout 1800
```

El límite es de 1800 s (30 min), no una duración esperada. Se realiza un
único intento y se guarda el registro, incluso si falla. No relanzar el
comando automáticamente. La finalización esperada es
`PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM`, todavía sin validación física.
Compartir `COMPILACION_CANON42.json` o el texto de finalización para continuar
con la auditoría. Si el IDE modifica metadatos generados, la comprobación
posterior preserva esa evidencia; no autoriza cambios del algoritmo.

La duración de los lotes físicos se acotará en el cliente. La transmisión
RAW de 56 ventanas por el protocolo anterior a 115200 baudios ronda nueve
minutos; cómputo, repeticiones y consultas añaden tiempo. Es una orientación
basada en capturas anteriores, no una medida del nuevo banco.

## Implicación científica

Esta etapa verifica una implementación canónica por condición, no toda la
variabilidad de las diez semillas ni generalización a otro motor. No
corrige por sí sola los límites ya observados: compromiso entre recall IR y
otras clases en 0 HP, rechazo de registros normales bajo cambio de carga,
variación FP32–INT8 y menor rendimiento de algunos selectores. Esos
resultados permanecen en el cierre exterior. El manuscrito no se modifica
durante esta preparación.
