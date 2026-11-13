# Validación de la compilación del banco TotalV2

La compilación aislada del primer intento completó los tres controles
previstos en 127.546 s, con retorno 0, sin timeout, errores ni advertencias.
Se verificaron los ELF de EntM, Masks y TotalV2 y las banderas de compilación.
No se repitió el entrenamiento de ninguno de los 32 representantes.

## Integridad de modelos e implementación

El ELF nuevo tiene SHA256
`85e8d7e2d1848833117fd4704493710cda0058fecedef3b379a80e9560171b1c`.
Los 32 modelos ocupan 113888 B y sus bytes, tamaños, SHA256 y FNV coinciden
con los artefactos exportados. También se contrastaron firmas, topología
`d--20--10--3`, tres capas Fully Connected, Softmax y tensores INT8
`[1,d] -> [1,3]`. Los descriptores enlazados ocupan 328 B cada uno según el
ABI ARM; su serialización de parámetros de 297 B se comprobó por separado,
sin confundir el padding del struct con el contenido del checksum.

Los dos ELF de control, EntM y Masks, son idénticos entre la compilación
aislada, el proyecto original D: y los respaldos. La revisión de instrucciones
del banco confirmó dos lecturas CYCCNT en la rama principal, una llamada a
cada una de las cuatro etapas y un perfil separado. La función de inferencia
no añade lecturas DWT internas. El escalado y la distancia conservan las
operaciones FP32 separadas y la suma secuencial; no se observaron FMA en
esas dos funciones ni opciones fast-math en las recetas revisadas.

No se introdujeron barreras DSB/ISB alrededor de CYCCNT. Esta revisión
comprueba el orden de instrucciones emitido y el acceso MMIO volatile;
no demuestra WCET ni sustituye las medidas físicas.

## Memoria del banco

| Medida | Bytes | Interpretación |
|---|---:|---|
| Flash por suma de secciones | 280332 | 53.4691 % de 512 KiB |
| Segmentos Flash programables | 280352 | Incluye 20 B de alineación |
| RAM por suma de secciones y reservas | 88460 | 67.4896 % de 128 KiB |
| Rango de direcciones RAM ocupado | 88464 | Incluye 4 B entre secciones |
| Reserva de stack | 4096 | Incluida en la cifra RAM anterior |
| Reserva de heap | 512 | Incluida en la cifra RAM anterior |
| Arena TFLM reservada | 16384 | Una sola arena reutilizada |

Los buffers verificados son una ventana de 32768 B y dos scratch de
16384 B, además de la arena de 16384 B. No se superponen. Quedan 42608 B
fuera del rango RAM reservado por el enlace. La ocupación corresponde a un
firmware de banco con 32 modelos, no a un despliegue final con uno solo.
El uso real de arena por modelo y la huella máxima de stack siguen pendientes
de comprobación física; las reservas y los archivos `.su` no los miden.

## Cambio generado por CubeIDE

El sello de preparación se conserva intacto. Al revisar sus 1968 entradas
después de compilar se encontró una única diferencia:
`.settings/language.settings.xml` de la copia aislada. CubeIDE generó los
atributos `env-hash` del detector de compilador y añadió TotalV2.

La comparación semántica verificó las quince configuraciones previas
ignorando exclusivamente ese atributo generado del proveedor identificado;
la configuración nueva conserva los proveedores de Masks salvo nombre,
identificador y dicho hash. Se guardaron el hash original y el posterior en
`AUDITORIA_ELF_TOTAL.json`. No se normalizó ni reemplazó el sello antiguo.
Los metadatos generados de la copia no se propagan sobre el proyecto D:.

## Paso siguiente

La instalación respaldada añade las fuentes, configuración y ELF auditado
al proyecto D:. La entrada de carga derivada `SinRecompilar` desactiva las
dos opciones de compilación previa para mantener el ELF revisado.

La primera puerta física comprobará una BASE, tres RUN principales y un
PROFILE para el representante 0 y la ventana global 568, más 17 rechazos
de protocolo. No se habilita todavía el selector de costo ni se incorporan
estos resultados estáticos como tiempos observados en el manuscrito.

Evidencias: `COMPILACION_TOTAL.json`, `AUDITORIA_ELF_TOTAL.json`,
`AUDITORIA_MAQUINA_TOTAL.json` y el informe de instalación, en
`despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/`.
