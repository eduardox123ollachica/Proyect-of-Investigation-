# Incidencias de ejecución

El primer intento del piloto E1 semilla 11 se ejecutó con el intérprete Python
aislado de Codex y se bloqueó durante el guardado `.keras`. El modelo parcial no
era un archivo Keras válido y se conserva aquí, separado de los resultados, como
registro de la incidencia.

La causa fue el uso combinado del intérprete aislado con los paquetes TensorFlow
del usuario. Todas las ejecuciones válidas se realizaron después con el Python
3.12 de Microsoft Store empleado originalmente por el proyecto.
