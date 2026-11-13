# Búsqueda interna RF/SVM CWRU V2

Plan:384 ajustes por rol (288RF y96SVM). Cuatro escenarios exteriores,
tres validaciones internas, ocho configuraciones por familia. RF usa11/22/33;
SVM se ajusta una sola vez por configuración/pliegue, sin triplicar semillas.
Ambos usan F14. El grid y su orden proceden del contrato original del piloto.
RF recibe float32 sin escalar; SVM usa scaler ajustado únicamente con train,
exportado a float32 y aplicado con resta/división explícitas. SVC mantiene
predict/votación nativa; su decision_function no sustituye esa regla.

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\clasicos_internos\ejecutar_clasicos.py" --run
```

Un proceso por ajuste, CPU/un hilo. Límites RF60s ySVM30s; presupuesto acumulado
3600s para procesos y comprobación individual. No incluye todo el preflight
o pausas entre comandos; es más conservador que contar solo subprocesos.
La parada se decide antes de iniciar otro ajuste, reservando su timeout
completo. La comprobación o terminación de un paso ya iniciado puede añadir
tiempo al alcanzar esa frontera; no es un límite exacto de reloj global.
Reserva orientativa30–60min, sin garantía. El presupuesto de esta carpeta
deriva de las16configuraciones medidas, sin repetir el piloto. No se amplían
límites ni se reintenta automáticamente. Fallos/interrupciones conservan
artefactos y detienen para auditoría. Guardado por modelo y paso con firmas.

El piloto fue solo temporal, confirmatory_fit=false y sin estimadores
persistidos; no se presenta como un resultado de búsqueda reutilizable.
Se mantienen los384ajustes por rol del protocolo, aunque algunos pares de
roles comparten entrenamiento; no se introduce una deduplicación nueva entre
pliegues. Esto no implica384adquisiciones independientes ni un test virgen.

Se guardan modelos joblib, parámetros, IDs, scaler SVM, predicciones,
probabilidades/vectores de decisión y métricas por clase. La reanudación
verifica modelos completos y firmas; un marcador único impide reiniciar el
presupuesto creando otra campaña. Un fallo necesita recuperación técnica
separada; no borrar locks o archivos para repetir. No se cargan modelos ajenos.

Al terminar imprime `CLASSICAL_INNER_COMPLETED: 384/384; RF288; SVM96`.
No necesita placa o COM. No selecciona configuraciones o máscaras, modifica
el detector, revela evaluación exterior ni edita el manuscrito. Las tasas
deficientes se conservan. Los tiempos PC no se presentan como latencia STM32.
