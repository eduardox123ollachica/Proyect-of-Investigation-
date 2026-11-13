# Recuperación de exportación T6 sin entrenar

F14 aprobó el piloto revisado (23.275771 s). T6 completó 50 épocas y 1250
actualizaciones, pero se detuvo la comprobación numérica del grafo congelado.
Ambos intentos y sus sellos permanecen intactos. El checkpoint de T6 permite
recuperar la exportación con **cero actualizaciones adicionales**; F14 no se repite.

La comprobación anterior comparaba la función congelada con lote 1 y el modelo
Keras ejecutado con lote 175. Esta revisión registra ambas comparaciones y exige
comparar también la función original y la congelada con el mismo lote 1, que
es la firma de inferencia exportada. Se conserva rtol=1e-6, atol=1e-7 y cero
cambios de clase entre ambas funciones a igual lote. No se relajan márgenes ni
se escogen ejemplos según sus predicciones. Si no cumple, se detiene de nuevo.

Se restauran los pesos y scaler guardados. No se ajusta ningún parámetro.
Se verifican sus hashes y se producen únicamente FP32/INT8, calibrando las 799
filas originales de entrenamiento, con diagnóstico en las mismas 175 ventanas.
El proceso tiene límite de 300 s, no abre 3 HP y no reintenta automáticamente.

El perfil T6 terminado será **compuesto**: tiempos previos al fallo más tiempos
de recuperación, incluyendo la importación y preparación adicional. No será
una medición de un único proceso ni una comparación justa de velocidad F14/T6.
Podrá usarse como escenario conservador de planificación, con ese límite
explícito; el costo operativo observado de todos los intentos se mantendrá aparte.
