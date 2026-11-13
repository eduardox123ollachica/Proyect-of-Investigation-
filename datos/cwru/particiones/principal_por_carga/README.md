# Partición principal CWRU por carga

## Alcance

Esta partición corresponde al subconjunto CWRU de 28 archivos formado por las
clases `N`, `IR` y `OR` centrada a las 6:00. Se utilizarán las dos posiciones de
acelerómetro disponibles en cada archivo:

- `DE`: acelerómetro del lado de accionamiento (*drive end*).
- `FE`: acelerómetro del lado del ventilador (*fan end*).

No deben describirse como ejes X/Y/Z. Son mediciones simultáneas en dos
ubicaciones del banco. Cada par `DE`–`FE` permanece unido durante todo el
experimento.

## Unidad de partición

La unidad experimental es el archivo `.mat` completo, no una ventana. Primero
se asigna cada archivo a un subconjunto y solo después se realiza el filtrado,
remuestreo y ventaneo. Por tanto, ninguna ventana derivada de una misma captura
puede aparecer en más de un subconjunto.

## División congelada

| Subconjunto | Cargas | Archivos | Uso permitido |
|---|---:|---:|---|
| Entrenamiento | 0 y 1 HP | 14 | Ajuste del modelo y de todo preprocesamiento aprendido |
| Validación | 2 HP | 7 | Selección de arquitectura, hiperparámetros y umbrales |
| Prueba | 3 HP | 7 | Evaluación final, una vez congeladas las decisiones |

Cada carga contiene un archivo normal, tres archivos IR —uno por diámetro de
falla— y tres archivos OR —uno por diámetro—. La prueba evalúa así una carga no
vista, pero conserva en sus fallas los tres diámetros representados durante el
entrenamiento. No es una prueba de generalización a diámetros no vistos.

## Relación con el preprocesamiento

La partición es independiente de la frecuencia de muestreo. Los archivos
locales proceden de la colección CWRU de 48 kHz y, en los 28 archivos, las
señales seleccionadas `DE` y `FE` tienen igual número de muestras. El protocolo
de preprocesamiento definirá por separado el filtrado antialias, el remuestreo,
la longitud de ventana y el solapamiento.

## Reglas para evitar fuga de información

1. `DE` y `FE` de una captura deben compartir siempre el mismo subconjunto.
2. El ventaneo se ejecuta después de consultar `manifest_particion.csv`.
3. Estadísticos de normalización, pesos de clase, cuantización representativa,
   centroides y umbrales se ajustan únicamente con entrenamiento.
4. Validación puede orientar decisiones; prueba no se utiliza para seleccionar
   variantes ni para ajustar umbrales.
5. Si se prueban ambas posiciones por separado o fusionadas, la comparación
   debe usar exactamente este mismo reparto de archivos.

## Irregularidades internas conocidas

El manifiesto fija explícitamente las variables que deben cargarse:

- `174_IR_0,014.mat` usa variables internas con prefijo `X173`.
- `175_IR_0,014.mat` contiene además una señal `X217_DE_time`; se ignora.
- `217_IR_0,021.mat` contiene señales `X215_*` duplicadas; se seleccionan
  `X217_DE_time` y `X217_FE_time`.
- `99_N_0,000.mat` contiene señales `X098_*` duplicadas; se seleccionan
  `X099_DE_time` y `X099_FE_time`.

Los `.mat` originales no se renombran, mueven ni modifican.

## Estado del procesado anterior

`datos/cwru/procesado/` contiene una ejecución histórica de un pipeline
unicanal `DE` a 2 kHz. No pertenece a este protocolo de dos posiciones a
12 kHz y no debe reutilizarse como entrada del nuevo experimento. Se conserva
solo como evidencia histórica hasta definir una ubicación de archivo adecuada.

