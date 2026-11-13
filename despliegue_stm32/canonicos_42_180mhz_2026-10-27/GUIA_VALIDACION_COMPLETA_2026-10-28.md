# Validación completa escalonada de los ocho modelos canónicos

Fecha: 28 de octubre de 2026, America/Lima.

El control inicial `20261028T050129_905025Z` fue auditado con estado
`PASS_GATE_CANON42_AUDITED_PENDING_FULL`. Se reconstruyeron 621 pares de
intercambios UART y 93 eventos, incluyendo 24 RUN, ocho PROFILE, dos BASE,
24 POST y tres rechazos de entradas inválidas. La auditoría confirmó la
procedencia del firmware y la coincidencia de las salidas con PC en los casos
comprobados. El uso observado de la arena fue 1188 bytes en los ocho modelos.

## Qué se completa

Se mantiene la validación escalonada elegida por el usuario. Los modelos,
semilla 42, épocas, parámetros y firmware son los mismos del control inicial.
El cliente reutiliza su evidencia auditada sin repetir esos casos.

| Cobertura | Ya comprobado en el gate | Nuevo en este lote | Total al completar |
|---|---:|---:|---:|
| Pares RAW modelo–ventana | 8 | 328 | 336 |
| RUN RAW, tres por par | 24 | 984 | 1008 |
| PROFILE RAW, uno por par | 8 | 328 | 336 |
| BASE, una por ventana | 2 | 54 | 56 |
| Pares POST modelo–ventana | 24 | 2736 | 2760 |

El banco RAW tiene 56 ventanas únicas: primera y última de los 28 MAT. Cada
modelo procesa las 42 ventanas de sus cargas de desarrollo. POST comprueba
las 1380 ventanas exteriores mediante sus características PC guardadas,
aplicando los dos modelos correspondientes a cada carga. Son 2760 pares;
POST realiza cero extracciones de señal.

Se envían 55 cargas RAW nuevas. La ventana gid0 ya cubrió sus modelos
autorizados. Para gid175 se conserva la BASE del gate y se ejecutan los
cuatro modelos que faltaban. Cada extracción nueva debe reproducir los
valores ARM de su BASE en las características activas; las inactivas deben
ser +0. Las diferencias numéricas respecto de la extracción PC se documentan
por separado y se exige la tolerancia acordada y coincidencia de bins
dominantes. Las salidas INT8 se comparan usando exactamente la misma entrada.

## Ejecución del usuario

Mantener la placa con el mismo firmware y cerrar las aplicaciones que ocupen
COM4. La carpeta de trabajo puede ser cualquiera: el comando usa rutas absolutas.
No hace falta compilar ni programar nuevamente la placa.

```powershell
$canon42 = "C:\Users\LENOVO\Desktop\Proyecto Investigación\despliegue_stm32\canonicos_42_180mhz_2026-10-27"

& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "$canon42\Validacion_Completa_PC\validar_canonicos_full.py" --full --port COM4 --max-seconds 1800
```

Duración orientativa: **17–20 minutos**, estimada a partir de las transmisiones
y tiempos del control inicial; no constituye una garantía. El presupuesto es
de **1800 segundos (30 minutos)**. Puede aparecer la advertencia habitual de
TensorFlow si resulta necesario comprobar una entrada INT8 nueva en PC.
El cliente no entrena ni cambia modelos. Los importes de tiempos del gate
pertenecen a estos mismos modelos canónicos; no se importan mediciones de
los antiguos representantes de calibración.

Se imprime una línea por par RAW completado y progreso cada 50 POST. Las
evidencias se guardan en `Validacion_Completa_PC/resultados/<marca UTC>/`.
La marca UTC del nombre lleva cinco horas de adelanto sobre Lima.

La finalización esperada es:

```text
CANON42_STAGED_FULL_PASS_PENDING_TRACE_AUDIT
```

Si falla o se interrumpe, conservar la carpeta indicada y compartir el mensaje
final. El cliente registra un intento único, conserva lo obtenido y no inicia
reintentos ni reanudaciones automáticas. No borrar el marcador de intento
para forzar otra ejecución.

## Interpretación y límites

PASS indica correspondencia entre implementaciones para los casos comprobados.
Una predicción puede coincidir con PC y ser incorrecta frente a la etiqueta.
El gate ya incluía tres casos de ese tipo; no son fallos de comunicación o de
portabilidad, ni permiten estimar la exactitud general con esos 24 ejemplos.

El estado final del cliente todavía requiere auditoría de las trazas y de los
conteos. La cobertura RAW es parcial y de desarrollo; POST cubre todo el
conjunto exterior solo desde las características PC. No se afirmará extracción
RAW exhaustiva del test. La latencia principal usa exclusivamente RUN RAW;
excluye UART, parser, checksums, selección, inicialización, BASE, PROFILE y
POST. Los percentiles son descriptivos del banco, no WCET ni tiempo de
adquisición a alerta. El reloj de 180 MHz procede de la configuración HAL
informada por el firmware; no se midió externamente en esta etapa.

La arena de 1188 bytes observada no es el uso total de RAM. Los 185388 bytes
de Flash y 88460 de RAM estática corresponden al banco de ocho modelos,
incluidas las reservas del enlazador. El stack máximo de este firmware sigue
sin medirse. El manuscrito no se modifica durante esta preparación.
