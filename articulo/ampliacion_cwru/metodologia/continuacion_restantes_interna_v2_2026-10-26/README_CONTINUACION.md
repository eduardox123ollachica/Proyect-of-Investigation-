# Continuación de todos los ajustes DNN internos pendientes

Auditoría:17 nuevos PASS,48 reutilizados,65/288 celdas. Quedan223 ajustes.
El último lote16 tomó175.265s; media10.954s por ajuste. Proyección de tiempos
contabilizados restantes:41–44min con este lote, sin garantía. Reservar
aproximadamente60–90min de reloj para incluir comprobaciones progresivas;
otras cargas del equipo y otras celdas pueden cambiar el tiempo.

Un solo comando encadena los pendientes sobre la campaña ya existente,
normalmente13 lotes de16 y uno de15. El ejecutor original y sus sellos se
conservan. Cada lote mantiene hasta16 ajustes y1800s; proceso300s; presupuesto
acumulado7200s, del que quedan116.24min. Verificaciones
compartidas entre lotes pueden añadir tiempo fuera del contador.

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\continuacion_interna\continuar_restantes.py" --run
```

No necesita placa ni COM. Guardado incremental por ajuste y lote. Se
detiene por fallo, interrupción, falta de progreso o presupuesto agotado;
no reintenta ni amplía límites. Al alcanzar288/288 imprime
`ALL_REMAINING_COMPLETED: 288/288 celdas DNN internas disponibles`.
Las48 importaciones y17 ajustes existentes no vuelven a entrenarse.

El controlador escribe un recibo adicional en
`resultados/cwru/ampliacion_v2/control_continuacion_interna_v2_2026-10-26/`.
Los modelos y lotes originales siguen en la misma campaña. Reejecutar este
comando permite continuar los pendientes completos verificados, salvo
incidencias que exijan auditoría. Conservar la salida final y ambas rutas.
La matriz DNN completa todavía no equivale a cerrar el detector, la búsqueda
clásica, la selección final ni la evaluación exterior. No se edita el manuscrito.
