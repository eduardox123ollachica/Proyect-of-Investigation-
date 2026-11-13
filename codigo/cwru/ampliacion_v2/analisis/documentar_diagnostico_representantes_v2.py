"""Informe y graficos del diagnostico ya cerrado; no abre datos fuente o modelos."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/representantes'))
from ejecutar_representantes import read,write,sha,verify_manifest

OUT=ROOT/'articulo/ampliacion_cwru/metodologia/diagnostico_representantes_v2_2026-10-25'


def main():
    entries=verify_manifest(OUT/'MANIFEST_DIAGNOSTICO_SHA256.json')
    report=read(OUT/'DIAGNOSTICO.json')
    normal=report['normal_record_summaries']
    jobs=report['jobs']
    f14={j['outer']:j for j in jobs if j['candidate']=='F14'}
    detectors={j['outer']:j for j in report['detector'] if j['candidate']=='F14'}
    cache=ROOT/'resultados/cwru/ampliacion_v2/cache_matplotlib_diagnostico'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'figure.dpi':140,'savefig.dpi':180})
    figures=OUT/'figuras';figures.mkdir(exist_ok=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for channel,index,color in [('DE',4,'#176899'),('FE',11,'#dd7f29')]:
        rows=sorted([r for r in normal if r['feature_index']==index],key=lambda r:r['load_hp'])
        xx=np.asarray([r['load_hp'] for r in rows],dtype=float)+(0 if channel=='DE' else .08)
        yy=np.asarray([r['median'] for r in rows])
        errors=np.asarray([[r['median']-r['minimum'] for r in rows],[r['maximum']-r['median'] for r in rows]])
        ax[0].errorbar(xx,yy,yerr=errors,fmt='o',capsize=4,color=color,label=channel)
    ax[0].set(xticks=range(4),xlabel='Carga del registro normal (HP)',ylabel='Frecuencia dominante (Hz)',
              title='Un registro normal por carga: mediana y rango')
    ax[0].legend();ax[0].grid(axis='y',alpha=.2)
    ratios=[detectors[o]['normal_median_to_threshold'] for o in range(4)]
    bars=ax[1].bar(range(4),ratios,color=['#798794','#c35538','#c35538','#c35538'],width=.6)
    ax[1].axhline(1,color='black',linestyle='--',linewidth=1,label='Umbral de rechazo')
    for bar,value in zip(bars,ratios):ax[1].text(bar.get_x()+bar.get_width()/2,value+.09,f'{value:.2f}×',ha='center')
    ax[1].set(xticks=range(4),xticklabels=['O0: val 1 HP','O1: val 0 HP','O2: val 0 HP','O3: val 0 HP'],
              ylabel='Mediana de distancia normal / umbral',title='Detector F14: distancia de normales etiquetados',ylim=(0,6.7))
    ax[1].tick_params(axis='x',labelsize=9);ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.2)
    fig.text(.5,.025,'Diagnóstico interno, semilla 11. Rangos de ventanas solapadas; no intervalos de confianza.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(figures/'normales_por_registro.png');fig.savefig(figures/'normales_por_registro.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,5.6))
    names=['DE energía','DE amplitud','FE RMS','FE asimetría','FE curtosis','FE energía','FE amplitud','FE entropía']
    counts=[report['clipped_values_per_original_feature'][i] for i in (3,5,7,8,9,10,12,13)]
    bars=ax[0].barh(range(8),counts,color='#176899')
    for bar,value in zip(bars,counts):ax[0].text(value+18,bar.get_y()+bar.get_height()/2,str(value),va='center',fontsize=9)
    ax[0].set(yticks=range(8),yticklabels=names,xlabel='Valores recortados, suma de 32 evaluaciones',
              title='Recortes de entrada INT8 por característica',xlim=(0,1800));ax[0].invert_yaxis()
    ax[0].grid(axis='x',alpha=.2)
    settings=[('numpy_original_macro_f1','FP32 original','#176899'),
              ('numpy_range_limited_macro_f1','FP32 + entrada limitada','#48a295'),
              ('numpy_input_quant_dequant_macro_f1','FP32 + cuant./decuant. de entrada','#d9a441'),
              ('saved_INT8_macro_f1','INT8 completo guardado','#c35538')]
    xx=np.arange(3)
    for pos,(key,label,color) in enumerate(settings):
        ax[1].bar(xx+(pos-1.5)*.19,[f14[o][key] for o in (1,2,3)],width=.18,label=label,color=color)
    ax[1].set(xticks=xx,xticklabels=['O1 / val 0 HP','O2 / val 0 HP','O3 / val 0 HP'],
              ylim=(0,1),ylabel='F1 macro',title='F14: intervención sobre entrada, pesos float fijos')
    ax[1].grid(axis='y',alpha=.2)
    ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=8,ncol=2)
    fig.text(.5,.025,'Reproducción NumPy validada frente a salidas guardadas. No es entrenamiento ni medición ARM.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,1));fig.savefig(figures/'recorte_y_reproduccion_numerica.png');fig.savefig(figures/'recorte_y_reproduccion_numerica.pdf');plt.close(fig)
    lines=['# Diagnóstico de los representantes V2','',
        '## Dictamen','',
        'No se encontró un error de orden de características, IDs, escalado o cuantización.',
        'La entrada INT8 y los scores del detector se reprodujeron exactamente en los32.',
        'El diagnóstico distingue dos limitaciones: transferencia de la representación normal',
        'entre registros/cargas y pérdida de información por el rango de entrada INT8.',
        'No se entrenó, reconvirtió, cambió un parámetro o calculó una predicción externa.','',
        'Sí se realizaron reproducciones numéricas: 22032 filas-forward NumPy con pesos',
        'guardados y score de normalidad con parámetros guardados. No se invocó TensorFlow',
        'o TFLite. El forward original coincidió en todas las clases con las referencias',
        f"guardadas; error máximo de probabilidades {report['numpy_probability_max_error']:.9g}.",
        'Eso respalda el diagnóstico, sin afirmar identidad bit a bit de kernels o probabilidades.','',
        '## 1. El registro normal de0HP tiene una representación distinta','',
        '| MAT / carga | Ventanas | RMS DE mediana | RMS FE mediana | Frecuencia dominante DE |',
        '|---|---:|---:|---:|---:|']
    for load in range(4):
        group={r['feature_index']:r for r in normal if r['load_hp']==load}
        lines.append(f"| {group[0]['file_id']} / {load}HP | {group[0]['windows']} | {group[0]['median']:.6f} | {group[7]['median']:.6f} | {group[4]['median']:.4f}Hz |")
    lines += ['',
        'En los tres F14 que validan0HP, las28normales se clasifican OR ya en FP32',
        'y no tienen entradas recortadas. Su error no se origina en saturación de entrada.',
        'La frecuencia dominante corresponde al mayor pico noDC; no es automáticamente la',
        'frecuencia de giro. Que NF12 rinda mejor en estos representantes es un indicio',
        'para evaluar esa representación ya predefinida, sin demostrar causalidad exclusiva',
        'de la frecuencia: son redes entrenadas por separado y solo hay una semilla.','',
        'Solo existe un MAT normal por carga. No puede separarse un efecto general de carga',
        'de una particularidad de ese registro, adquisición, montaje o ruido. La reconstrucción',
        'desde MAT ya se auditó anteriormente; este diagnóstico no repitió extracción RAW.',
        'No hay evidencia instrumental nueva para cambiar Fs, etiquetar MAT97 como falla o eliminarlo.','',
        '![Normales por registro](figuras/normales_por_registro.png)','',
        '## 2. El recorte de entrada explica buena parte de la degradación INT8 en F14','',
        'De4516valores recortados,4218(93.40%) son FE RMS, energía y amplitud.',
        'Las frecuencias dominantes4/11 no se recortan. Se contabilizan valores de',
        'características en evaluaciones repetidas, no4516ventanas físicas diferentes.',
        'En F14 hay120cambios FP32–INT8 al validar0HP;119ocurren en ventanas con recorte.',
        'Globalmente35de486cambios ocurren sin recorte: también intervienen redondeo,',
        'cuantización de pesos/activaciones y decisiones cercanas a las fronteras.','',
        'Con los mismos pesos float se comparan entradas originales, entradas limitadas',
        'al rango representable y entradas cuantizadas/decuantizadas. Son intervenciones',
        'diagnósticas, no modelos candidatos nuevos. El límite continuo y el recorte',
        'después de redondear difieren alrededor de medio paso de cuantización.','',
        '| F14 outer | Float original | Solo límite de entrada | Cuant./decuant. entrada | INT8 guardado |',
        '|---|---:|---:|---:|---:|']
    for outer in (1,2,3):
        row=f14[outer]
        lines.append(f'| {outer} | '+' | '.join(f'{row[key]:.6f}' for key,_,_ in settings)+' |')
    lines += ['',
        'Limitar la entrada basta para reproducir la caída en gran parte de estos casos.',
        'No se atribuye el residuo entre entrada cuant./decuant. con pesos float e INT8',
        'exclusivamente a pesos: también incluye activaciones internas y aritmética.',
        'NF12 tiene los mismos recortes que F14 en estos scopes y distinta sensibilidad;',
        'por ello recorte presente no implica idéntica degradación para cualquier red.','',
        '![Cuantización y reproducción](figuras/recorte_y_reproduccion_numerica.png)','',
        '## 3. El detector aprende una normalidad demasiado estrecha para estos registros','',
        '| F14 outer / validación | Umbral | Mediana distancia normal / umbral | Normales aceptadas |',
        '|---|---:|---:|---:|']
    for outer in range(4):
        row=detectors[outer]
        lines.append(f"| {outer} / {row['inner']}HP | {row['threshold']:.3f} | {row['normal_median_to_threshold']:.3f} | {row['normal_accepted']}/{row['normal_windows']} |")
    lines += ['',
        'RMS, energía y amplitud de DE/FE aportan92.95–96.38% de la distancia media',
        'de normales0HP para F14. Las frecuencias ya están excluidas del detector:',
        'F14 y NF12 comparten exactamente sus parámetros. No basta quitarlas de la DNN',
        'para corregir el rechazo del detector.',
        'El scaler normal-only estima variación de un solo MAT; las diferencias entre',
        'registros pueden equivaler a muchas desviaciones de esa normalidad estrecha.',
        'La calibración acepta56/58normales de su propio MAT, sin garantizar cobertura',
        'de otra carga. Outer2/3 comparten ajuste y scores normales0HP, pero otro registro',
        'de calibración cambia el umbral986.174→337.697. No se alteró ninguno.','',
        '## 4. Consecuencias para el plan y el artículo','',
        'Aumentar solo50→100épocas no aborda los26ajustes ya detenidos por paciencia.',
        'Los resultados respaldan separar el estudio de convergencia de los problemas',
        'de representación, cuantización y normalidad. La pérdida total incluye L2:',
        'el JSON separa CE y penalización en el mejor checkpoint, sin inventar la',
        'descomposición de una curva que no se guardó por término.','',
        'El detector actual mide distancia respecto al dominio normal observado. Los',
        'rechazos de normales impiden presentarlo como confirmación fiable de falla',
        'entre todas estas condiciones. Esto es una limitación experimental sustentada,',
        'no una razón para relabelar normales o aumentar el umbral con validación.','',
        'La próxima decisión debe documentarse como suplemento de desarrollo antes',
        'de nuevas ejecuciones: mantener la referencia original, evaluar la robustez',
        'de las máscaras predefinidas con las semillas/cargas completas y, si se añade',
        'una variante de compresión de amplitud, fijar fórmula/índices y train-only antes.',
        'No se elige NF12 automáticamente con estos32, no se recalibra INT8 con validación',
        'ni se usa N0 para corregir retrospectivamente su propio umbral.',
        'Para el detector, ninguna variante puede prometer resolver la falta de registros',
        'normales independientes; debe evaluarse especificidad por MAT/carga y reportar',
        'los fallos, incluso si ello limita su papel a rechazo de dominio.',
        'Los32se conservan como referencia y representantes técnicos. Si se enmienda',
        'el procedimiento principal, los nuevos representantes deberán provenir de la',
        'misma versión seleccionada: no mezclar métricas o costos de modelos distintos.',
        'El costo total STM32 sigue pendiente y no se modifica el manuscrito en esta etapa.','',
        '## Fuentes y límites','',
        '- [LiteRT: cuantización entera](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant): el conjunto representativo estima rangos; se cuantizan pesos y activaciones.',
        '- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): estadísticos de entrenamiento reutilizados y sensibilidad a valores extremos.',
        '- [CWRU: adquisición](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures): posiciones DE/FE y configuraciones de muestreo documentadas.',
        '- Evidencia local: `DIAGNOSTICO.json`, cinco CSV, cierre32 y auditoríaCWRU previa. Todos los datos son retrospectivos y ya explorados.',
        'No se calculan p-valores o intervalos IID a partir de ventanas solapadas.',
        'Los gráficos son recursos diagnósticos exportables; todavía no se insertan en el artículo.','']
    report_path=OUT/'INFORME_DIAGNOSTICO.md'
    with report_path.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    paths=list(figures.iterdir())+[Path(__file__),report_path,OUT/'MANIFEST_DIAGNOSTICO_SHA256.json',OUT/'MANIFEST_DIAGNOSTICO_SHA256.sha256']
    entries.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    manifest=OUT/'MANIFEST_INFORME_SHA256.json';write(manifest,entries)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(f'Informe y4graficos PNG/PDF: PASS; {len(entries)}recursos sellados.')


if __name__=='__main__':main()
