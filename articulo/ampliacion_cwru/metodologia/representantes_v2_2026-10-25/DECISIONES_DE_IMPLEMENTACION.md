# Decisiones de esta preparación

1. Se añade un suplemento limitado a los 32 representantes. Se conservan los
   documentos sellados y sus banderas históricas; no se declara congelado todo
   el protocolo ni se habilita evaluación externa.
2. Se conserva el objetivo de costo total confirmado por el usuario. El ejecutor
   no importa el selector COST y deja la latencia STM32 en null.
3. Se entrega primero un lote de uno para verificar el recorrido real y sus
   artefactos antes de los otros 31. Cada continuación la inicia el usuario.
4. Se fijan 300 s por worker y 7200 s acumulados por campaña como decisiones
   operativas basadas en pilotos. No se interpretan como ETA o WCET.
5. La firma incluye validación interna además de ajuste. Los representantes
   outer0/1 tienen el mismo ajuste 2/3 HP, pero distinta validación y ES; se
   conservan como ajustes diferentes.
6. El almacenamiento histórico no decide los roles V2. Las filas se eligen con
   los manifiestos; se declara el mapeo de cachés que pueden incluir filas
   excluidas, sin prometer una traza de accesos del sistema operativo.
7. Se reutiliza la aritmética del detector ya ensayada. Para obtener paquetes
   autónomos se permite recomputarla en cada representante, sin multiplicación
   por semillas DNN ni interpretación independiente del alias F14/NF12.
8. Se mantiene el entorno determinista de los pilotos, incluyendo oneDNN
   desactivado. El ensayo sintético inicial con el valor predeterminado queda
   preservado y no se usa como resultado científico.
9. El guardado incremental no permite reanudar un entrenamiento parcial desde
   una época intermedia: faltaría restaurar estado del optimizador/RNG. Ante
   fallo se detiene la cola; se auditará recuperación de conversión desde el
   checkpoint si procede, evitando un reentrenamiento automático.
10. Los bancos generados contienen referencias PC de características. La prueba
    física posterior debe seguir usando señales/ventanas trazables del banco
    autorizado para medir también extracción; no basta enviar características
    para declarar costo del pipeline completo.

Todos los archivos del ejecutor son nuevos. Solo se actualiza el documento
operativo mutable `despliegue_stm32/ESTADO_ACTUAL_ENTREGA_CWRU.md` para continuidad.
No se cambiaron datos, particiones, fuentes selladas anteriores, modelos
históricos, manuscrito o proyecto del disco D.
