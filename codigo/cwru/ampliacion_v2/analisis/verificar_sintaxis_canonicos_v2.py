"""Comprueba cinco unidades ARM reales sin objetos, enlazado, COM o build.

Lee las opciones generadas de la configuración SR anterior, conserva opciones
semánticas y reemplaza únicamente includes privados, entrada y salidas de
compilación por -fsyntax-only. La preparación se autentica antes y después.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
SCRIPT=Path(__file__).resolve()
PREPARER=ROOT/'codigo/cwru/ampliacion_v2/canonicos_stm32/preparar_proyecto.py'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module():
    spec=importlib.util.spec_from_file_location('canon42_syntax_preparer',PREPARER)
    require(spec is not None and spec.loader is not None,'Preparador no disponible')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def flags_from_make(path,compiler,project):
    candidates=[line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines()
        if line.lstrip().startswith(compiler+' ')]
    require(candidates,'No hay comando '+compiler+' en '+str(path))
    original=candidates[0]
    tokens=re.findall(r'"[^"\r\n]*"|\S+',original)
    require(tokens[:2]==[compiler,'"$<"'],'Entrada make inesperada')
    result=[];removed=[];skip_next=False
    for token in tokens[2:]:
        if skip_next:removed.append(token);skip_next=False;continue
        if token=='-o':removed.append(token);skip_next=True;continue
        if token in ('-c','-MMD','-MP','-fstack-usage','-fcyclomatic-complexity') or token.startswith(('-MF','-MT')):
            removed.append(token);continue
        if token.startswith('-I../'):
            relative=token[5:]
            if relative=='CWRU_TotalV2':relative='CWRU_Canon8'
            target=(project/relative).resolve()
            require(target.is_relative_to(project.resolve()) and target.is_dir(),'Include fuera de la copia o inexistente')
            result.append('-I'+str(target));continue
        require('"' not in token and '$' not in token,'Token make inesperado: '+token)
        require(not token.startswith(('-I','-o','-MF','-MT')),'Opción de salida/include no normalizada')
        result.append(token)
    require(not skip_next and '-c' not in result and not any(t.startswith(('-MF','-MT')) for t in result),'Salidas habilitadas')
    return result+['-fsyntax-only'],dict(original=original,removed_output_options=removed,
        include_replacement='CWRU_TotalV2 -> CWRU_Canon8; all paths resolve inside isolated copy')


def main():
    p=module();report_path=p.OUT/'VERIFICACION_SINTAXIS_ARM.json'
    require(not report_path.exists(),'No sobrescribir verificación existente')
    prep=p.verify();manifest_before=sha(p.MANIFEST)
    tools,_=p.toolchain()
    make_private=p.SOURCE/p.BASE/'CWRU_TotalV2/subdir.mk'
    make_src=p.SOURCE/p.BASE/'Src/subdir.mk'
    make_hashes={str(path):sha(path) for path in (make_private,make_src)}
    cases=[('private_pipeline',p.PROJECT/p.PRIVATE/'cwru_total.c','arm-none-eabi-gcc',make_private),
        ('private_catalog',p.PROJECT/p.PRIVATE/'cwru_total_catalog.cpp','arm-none-eabi-g++',make_private),
        ('private_neural_network',p.PROJECT/p.PRIVATE/'cwru_total_nn.cpp','arm-none-eabi-g++',make_private),
        ('real_UART_router',p.PROJECT/'Src/uart_commands.c','arm-none-eabi-gcc',make_src),
        ('real_UART_POST_integration',p.PROJECT/'Src/cmd_cwru_feat.c','arm-none-eabi-gcc',make_src)]
    records=[];start=time.monotonic()
    for label,source,compiler,makefile in cases:
        options,derivation=flags_from_make(makefile,compiler,p.PROJECT)
        executable=tools/(compiler+'.exe');command=[str(executable),str(source),*options]
        row=dict(unit=label,source=str(source),source_sha256=sha(source),command=command,
            compiler_sha256=sha(executable),makefile=str(makefile),makefile_sha256=sha(makefile),derivation=derivation)
        t=time.monotonic()
        try:
            process=subprocess.run(command,cwd=p.ROOT,stdin=subprocess.DEVNULL,capture_output=True,
                text=True,encoding='utf-8',errors='replace',timeout=60,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            row.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,
                passed=process.returncode==0 and not re.search(r'\b(?:warning|error):',process.stdout+process.stderr,re.I))
        except Exception as exc:
            row.update(returncode=None,stdout='',stderr='',passed=False,error=type(exc).__name__+': '+str(exc))
        row['seconds']=time.monotonic()-t;records.append(row)
        print(label+': '+('PASS' if row['passed'] else 'FAIL'),flush=True)
    intact=False;error=None
    try:
        p.verify();require(sha(p.MANIFEST)==manifest_before,'Sello de preparación cambió')
        require(all(sha(path)==digest for path,digest in make_hashes.items()),'Makefiles de origen cambiaron')
        intact=True
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    passed=intact and len(records)==5 and all(row['passed'] for row in records)
    report=dict(status='PASS_SYNTAX_PENDING_FULL_BUILD_AND_ARM' if passed else 'FAIL_SYNTAX_REQUIRES_REVIEW',
        units=records,units_passed=sum(row['passed'] for row in records),units_total=5,
        elapsed_seconds=time.monotonic()-start,script_path=str(SCRIPT),script_sha256=sha(SCRIPT),
        preparer_sha256=sha(PREPARER),preparation_manifest_sha256=manifest_before,
        preparation_authenticated_before=True,preparation_authenticated_after=intact,integrity_error=error,
        configuration=prep['configuration'],project=str(p.PROJECT),
        object_files_generated=False,linked_firmware=False,full_build_executed=False,
        COM_opened=False,hardware_flashed=False,D_files_modified=False,
        limitation='ARM syntax/type checking of five real translation units; not linking, execution, memory or UART validation')
    p.write(report_path,report)
    print(json.dumps(dict(status=report['status'],report=str(report_path),report_sha256=sha(report_path)),ensure_ascii=True))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
