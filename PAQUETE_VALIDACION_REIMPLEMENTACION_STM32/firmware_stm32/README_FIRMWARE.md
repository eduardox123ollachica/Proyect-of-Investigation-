# Evidencia de firmware STM32

## Proyectos incluidos

| Carpeta del paquete | Ruta histórica en el disco D | Función |
|---|---|---|
| `E1_Prueba_2` | `D:\STM32IDE\Prueba 2` | Firmware canónico E1 |
| `E4_Prueba_3` | `D:\STM32IDE\Prueba 3` | Firmware canónico E4 |
| `B4_Prueba_2_Recursos` | `D:\STM32IDE\Prueba 2 Recursos` | Build controlado para atribución de memoria y stack |

No se incluyó `D:\STM32IDE\Prueba 4` porque corresponde a la fase CWRU y no respalda las afirmaciones del artículo actual.

## Archivos conservados por proyecto

- `Src/` e `Inc/`;
- `Startup/`;
- `.project`, `.cproject`, `.settings/` y archivos `.launch`;
- scripts de enlazado;
- ELF, MAP y LIST históricos;
- archivos `makefile`, `*.mk` y `*.su` conservados en `Debug/`.

Los objetos `*.o` y dependencias `*.d` no se copiaron porque son productos intermedios redundantes. Los binarios, mapas, listados, fuentes y opciones de compilación sí se conservaron.

## Dependencias compartidas

`dependencias_compartidas/` contiene una sola copia de `Drivers`, `Lib`, `Middlewares` y `TFLM` tomada del proyecto E1. Los manuales CHM se omitieron porque no intervienen en la compilación. La auditoría previa comprobó que E1 y E4 comparten la misma configuración; las dependencias se deduplicaron para reducir el tamaño del ZIP.

Para volver a abrir un proyecto en STM32CubeIDE, ejecutar `PREPARAR_PROYECTO_STM32.ps1` con `E1`, `E4` o `B4`. El script copia las dependencias compartidas dentro de la carpeta seleccionada y no sobrescribe una dependencia que ya exista.

## Correspondencia canónica

- E1: `g_model[]` coincide con el TFLite canónico de 4480 bytes, SHA-256 `C21EE1E33A22B2A04CC396F5C2F02089D664CE8DA30173A7AB67AD9B974E985D`.
- E4: `g_model[]` coincide con el TFLite canónico de 4480 bytes, SHA-256 `C505A32CC620965A4112596D5548E5C61143AE1CA729BF7CB7D9A1E3B50D5ADD`.

## Archivos que deben conservarse si se actualiza manualmente desde D

Si se reemplaza alguno de los proyectos con una versión más reciente, copiar la carpeta completa solo después de confirmar que corresponde a la versión que produjo los resultados. Como mínimo deben conservarse `Src`, `Inc`, `.settings`, `.project`, `.cproject`, los linker scripts y los archivos ELF/MAP/LIST de `Debug`.

