"""Prueba breve integral con doce filas sinteticas; no lee CWRU ni modelos previos."""
import os
import tempfile
from pathlib import Path

from ejecutar_representantes import ENVIRONMENT, PREP, ROOT, source_hashes, versions, write


def main():
    # Las variables se fijan antes de importar bibliotecas numericas/TF.
    for key,value in ENVIRONMENT.items():
        if key != 'PYTHONHASHSEED':os.environ[key]=value
    import numpy as np
    from numerica_representantes import normalize_f32, metrics3
    from worker_representantes import configure_tf, fit_network, export_models, evaluate_lite, weights_hash
    tf,keras=configure_tf(11)
    rng=np.random.default_rng(123)
    x=rng.normal(size=(12,6)).astype(np.float32)
    y=np.asarray([0,1,2]*4,dtype=np.int64)
    validation=rng.normal(size=(6,6)).astype(np.float32)
    val_y=np.asarray([0,1,2]*2,dtype=np.int64)
    events=[]
    config=dict(l2_hidden=.001,learning_rate=.005,batch_size=32,max_epochs=3,
                early_stopping=dict(patience=10,minimum_improvement=1e-12))
    temp_base=ROOT/'resultados/cwru/ampliacion_v2/pruebas_temporales_representantes'
    temp_base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cwru_v2_synthetic_',dir=temp_base) as tmp:
        directory=Path(tmp)
        model,state=fit_network(tf,keras,x,y,validation,val_y,config,11,directory,
                               lambda stage,**extra:events.append(dict(stage=stage,**extra)))
        assert state['epochs_executed']==3 and state['optimizer_updates']==3
        with np.load(directory/f"checkpoint_epoca_{state['best_epoch']:02d}.npz",allow_pickle=False) as best:
            assert all(np.array_equal(w,best[f'weight_{i:02d}']) for i,w in enumerate(model.get_weights()))
        assert (directory/'estado_antes_conversion.json').is_file()
        fp,integer,original,export=export_models(tf,model,x,validation,directory,
                                               lambda stage,**extra:events.append(dict(stage=stage,**extra)))
        p,_,_,fp_details=evaluate_lite(tf,fp,validation,False)
        iq,_,inputs,iq_details=evaluate_lite(tf,integer,validation,True)
        assert inputs.dtype==np.int8 and export['representative_rows']==12
        assert weights_hash(model)==state['model_weights_sha256']
        metrics=metrics3(val_y,np.argmax(iq,axis=1))
        report=dict(status='PASS',versions=versions(),source_sha256=source_hashes(),CWRU_arrays_opened=False,
            synthetic_training_rows=12,synthetic_validation_rows=6,epochs=3,optimizer_updates=3,
            best_checkpoint_restoration_verified=True,checkpoint_written_before_conversion=True,
            graph_trace_updates=0,full_INT8_verified=True,freeze=export['freeze'],
            representative_rows=12,fp32_details=fp_details,int8_details=iq_details,
            synthetic_macro_f1_not_project_result=metrics['macro_f1'],events=events)
    PREP.mkdir(parents=True,exist_ok=True)
    write(PREP/'PRUEBA_TENSORFLOW_SINTETICA.json',report)
    print('PASS: tres updates sinteticos, restauracion, congelacion batch1 y FP32/INT8. CWRU: cero ajustes.')


if __name__=='__main__':main()
