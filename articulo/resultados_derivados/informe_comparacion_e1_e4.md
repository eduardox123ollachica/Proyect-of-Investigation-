# Comparación derivada E1-E4

Fecha de generación: 2026-10-11.

## Procedencia

El análisis usa exclusivamente las 1332 filas alineadas de:

- `resultados/arciniegas/e1/validacion_rawall/rawall_todas.csv`
- `resultados/arciniegas/e4/validacion_rawall/rawall_todas.csv`

Las claves `indice_test`, `file`, `window` y `label_real` coinciden en todas las
filas. Por tanto, la comparación es pareada sobre las mismas ventanas.

## Resultados principales

- E1: accuracy 97.8979 % y macro-F1 97.8089 %.
- E4: accuracy 99.9249 % y macro-F1 99.9305 %.
- Diferencia E4-E1: +2.0270 puntos porcentuales.
- Ambos modelos acertaron 1303 ventanas.
- E4 corrigió 28 ventanas erradas por E1.
- E4 introdujo un error en una ventana acertada por E1.
- No hubo ventanas erradas simultáneamente por ambos modelos.

E1 concentró sus 28 errores en ventanas de VELOCIDAD MEDIA clasificadas como
DETENIDO. E4 clasificó correctamente las 360 ventanas MEDIA y cometió un único
error: una ventana de VELOCIDAD ALTA fue clasificada como DETENIDO.

## Comparación pareada

El test exacto de McNemar sobre las ventanas discordantes produjo
`p = 1.1175871e-7`. Este resultado indica una asimetría clara: E4 corrige muchas
más decisiones de E1 de las que deteriora.

Las ventanas de un mismo archivo se solapan y no son independientes. Por ello,
McNemar a nivel de ventana se conserva como análisis complementario. Como análisis
más conservador se realizó un bootstrap pareado, estratificado por clase y agrupado
por los 37 archivos de prueba, con 10 000 iteraciones y semilla 20261011. El
intervalo percentil del 95 % para la mejora de accuracy fue de +0.5255 a +3.8288
puntos porcentuales.

## Interpretación y límite

Los dos análisis respaldan que E4 supera al modelo E1 actualmente desplegado. Sin
embargo, comparan dos entrenamientos concretos y no demuestran por sí solos que la
ventaja sea estable frente a todas las inicializaciones. Esa cuestión corresponde
al experimento multisemilla definido en
`articulo/metodologia/protocolo_robustez_multisemilla.md`.

## Reproducibilidad

Ejecutar:

```powershell
python articulo/analisis/generar_analisis_e1_e4.py
```

El script regenera las métricas, matrices, figuras SVG y resultados estadísticos.
