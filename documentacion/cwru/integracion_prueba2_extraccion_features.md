# Integración de la prueba aislada de características CWRU en `Prueba 2`

## Objetivo y alcance

Esta prueba valida solamente la primera parte del pipeline: recibir una ventana dual DE--FE, extraer 14 características en el STM32 y compararlas con las características congeladas del PC. No ejecuta todavía la DNN ni el detector normal-only.

No es posible enviar los 8192 valores de una ventana en un único comando `RAWFEAT`: la línea superaría ampliamente los 1024 bytes del buffer de `main.c`. Por ello, `cmd_cwru_feat.c` implementa una transferencia ASCII por bloques con confirmación de cada bloque.

## Archivos que se agregan a `Prueba 2`

Copiar a `D:\STM32IDE\Prueba 2\Src\`:

- `modelos/cwru/canonico_seed42/firmware/cwru_feature_extraction.c`;
- `modelos/cwru/canonico_seed42/firmware/cmd_cwru_feat.c`.

Copiar a `D:\STM32IDE\Prueba 2\Inc\`:

- `modelos/cwru/canonico_seed42/firmware/cwru_feature_extraction.h`;
- `modelos/cwru/canonico_seed42/firmware/cwru_pipeline_config.h`.

Los archivos históricos `feature_extraction.cpp` y `feature_extraction.h` pueden permanecer en el proyecto: sus símbolos corresponden al pipeline anterior de 16 muestras y no colisionan con los nuevos nombres CWRU.

## Archivo existente que debe modificarse

### `Inc/uart_command_handlers.h`

Agregar el prototipo junto a los demás comandos:

```c
void CmdCwruFeat_Execute(
    char *command,
    char *response,
    size_t response_capacity
);
```

### `Src/uart_commands.c`

Agregar esta rama antes del `else` final que responde `RECIBIDO`:

```c
else if (strncmp(comando, "CWRUFEAT_", 9) == 0)
{
    CmdCwruFeat_Execute(
        comando,
        respuesta,
        capacidad_respuesta
    );
}
```

No es necesario modificar `main.c`: las tramas creadas por el script permanecen por debajo de su buffer de recepción de 1024 bytes y la respuesta de 14 características cabe en sus 512 bytes. El contador DWT ya se inicializa allí.

## Configuración de STM32CubeIDE

Después de copiar los archivos, refrescar el proyecto para que CubeIDE incluya los dos nuevos `.c` en la compilación. La versión actual de `Prueba 2` ya compila `CMSIS-DSP/Source/TransformFunctions` y `CommonTables`, contiene la RFFT de 4096 puntos y enlaza `libm` y soporte de `printf` en coma flotante. No se necesita añadir una biblioteca distinta.

Los arreglos del comando son estáticos y consumen aproximadamente 64 KiB adicionales de SRAM: 32 KiB para la ventana dual y 32 KiB para entrada y salida de la RFFT. Debe revisarse el archivo MAP después de compilar. No deben trasladarse estos arreglos al stack.

## Protocolo UART

1. `CWRUFEAT_BEGIN` reinicia la recepción.
2. `CWRUFEAT_DATA,<offset>,<valores...>` carga un bloque a partir del offset indicado.
3. `CWRUFEAT_RUN` calcula y devuelve:

```text
CWRUFEAT_RESULT,<ciclos>,<14 características>
```

El orden transmitido es `DE[0], FE[0], DE[1], FE[1], ...`. El script Python construye este orden automáticamente a partir de `X_test.npy`.

## Ejecución desde Python

Con la placa programada y sin otro monitor serie abierto:

```powershell
python codigo/cwru/validacion_stm32/validar_features_cwru_stm32.py --port COM4 --index 0
```

Puede repetirse con índices diferentes. Por ejemplo:

```powershell
python codigo/cwru/validacion_stm32/validar_features_cwru_stm32.py --port COM4 --index 120
```

El script transmite la señal cruda, recibe las características calculadas por el STM32 y las compara con `features_14/X_features_test.npy`. Los CSV y el resumen JSON quedan en `resultados/cwru/validacion_stm32_features/`.

Esta transferencia a 115200 baudios es deliberadamente offline. No constituye evidencia de adquisición dual en tiempo real.
