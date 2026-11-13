"""Plan acotado de tiempos RF/SVM/detector; no ajusta modelos ni lee matrices."""
from __future__ import annotations
from itertools import product


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_jobs(protocol):
    order = protocol['candidate_order']
    require(order == ['F14','T6','S8','NF12','NM10','NE12','DE7','FE7'], 'Catalogo distinto')
    rf, svm = protocol['classical_comparators']['RF'], protocol['classical_comparators']['SVM']
    require(rf['n_estimators']==[100,300] and rf['max_depth']==[None,12]
            and rf['min_samples_leaf']==[1,3] and rf['n_jobs']==1, 'Grid RF distinto')
    require(svm['C']==[.1,1,10,100] and svm['gamma']==['scale',.01]
            and svm['probability'] is False, 'Grid SVM distinto')
    jobs=[]
    for i,(trees,depth,leaf) in enumerate(product(rf['n_estimators'],rf['max_depth'],rf['min_samples_leaf'])):
        jobs.append(dict(job_id=f'RF_{i:02d}',family='RF',config_index=i,timeout_seconds=60,
            feature_mask='F14',params=dict(n_estimators=trees,max_depth=depth,min_samples_leaf=leaf,
            max_features=rf['max_features'],criterion=rf['criterion'],bootstrap=rf['bootstrap'],
            class_weight=rf['class_weight'],n_jobs=1,random_state=11)))
    for i,(c,gamma) in enumerate(product(svm['C'],svm['gamma'])):
        jobs.append(dict(job_id=f'SVM_{i:02d}',family='SVM',config_index=i,timeout_seconds=30,
            feature_mask='F14',params=dict(C=c,gamma=gamma,kernel='rbf',class_weight='balanced',
            probability=False,tol=.001,cache_size=200,shrinking=True,max_iter=-1,
            decision_function_shape='ovr',break_ties=False)))
    indices={name:[i for i in protocol['candidates'][name] if i not in (4,11)] for name in order}
    unique,aliases=[],{}
    seen={}
    for name in order:
        key=tuple(indices[name])
        require(key and len(set(key))==len(key),'Detector sin columnas o duplicadas')
        if key in seen:
            aliases[name]=seen[key]
        else:
            seen[key]=name; unique.append(name)
    require(aliases=={'NF12':'F14'} and len(unique)==7,'Representaciones detector distintas')
    jobs.append(dict(job_id='DETECTOR',family='DETECTOR',timeout_seconds=60,
        candidate_order=order,indices_by_candidate=indices,unique_candidates=unique,
        aliases=aliases,repetitions=3,maximum_unique_fit_calls=21,
        normal_fit_loads=[1],normal_calibration_loads=[2],normal_fit_windows=58,
        normal_calibration_windows=58,refit_after_calibration=False))
    require(len(jobs)==17 and sum(j['timeout_seconds'] for j in jobs)==780,'Presupuesto de procesos distinto')
    return jobs


def validate_result(result,job):
    require(isinstance(result,dict) and result.get('status')=='PASS'
            and result.get('job_id')==job['job_id'] and result.get('family')==job['family'],
            'Resultado de trabajo incompleto o ajeno')
    require(result.get('training_windows')==799 and result.get('validation_windows')==175
            and result.get('no_load3_arrays_opened') is True
            and result.get('metrics_used_for_selection') is False
            and result.get('confirmatory_fit') is False,'Alcance de resultado distinto')
    if job['family'] in ('RF','SVM'):
        require(result.get('config_index')==job['config_index'] and result.get('requested_params')==job['params'],
                'Configuracion distinta del plan')
    else:
        require(result.get('fit_windows')==58 and result.get('calibration_windows')==58
                and result.get('unique_representations')==7 and len(result.get('rows',[]))==21
                and result.get('aliases')=={'NF12':'F14'},'Detector incompleto o repetido por alias')
    return result
