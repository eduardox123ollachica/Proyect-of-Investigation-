"""Valida con CubeIDE real en workspace aislado; no conecta ni programa hardware."""
from pathlib import Path
import subprocess
import json
import os
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/correccion_cubeide_2026-10-22'
IDE=Path(r'D:\STM32CubeIDE_2.2.0\STM32CubeIDE')
NAME='Prueba_2_CWRU_Extraccion'


def main():
    workspace=OUT/'workspace_aislado'
    if workspace.exists():raise FileExistsError('Usar una validación nueva; no sobrescribir logs')
    project=OUT/'prueba_ide'/NAME
    args=[str(IDE/'stm32cubeidec.exe'),'--launcher.suppressErrors','-nosplash',
          '-application','org.eclipse.cdt.managedbuilder.core.headlessbuild',
          '-data',str(workspace),'-import',str(project),'-no-indexer',
          '-build',NAME+'/Debug','-build',NAME+'/Release']
    env=dict(os.environ)
    tool=next((IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    make=next((IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin'))
    env['PATH']=str(tool)+os.pathsep+str(make)+os.pathsep+env.get('PATH','')
    start=time.monotonic()
    with (OUT/'CUBEIDE_HEADLESS.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(args,env=env,cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 encoding='utf-8',errors='replace')
        for line in process.stdout:
            log.write(line);log.flush()
            if any(s in line.lower() for s in ('building','build of','error','warning','finished','import','size','text','data','bss')):
                print(line.strip()[:300],flush=True)
        code=process.wait(timeout=30)
    elfs={c:str(project/c/(NAME+'.elf')) for c in ('Debug','Release') if (project/c/(NAME+'.elf')).is_file()}
    report=dict(returncode=code,elapsed_seconds=time.monotonic()-start,elfs=elfs,
                ide_real_headless=True,workspace_isolated=True,user_workspace_modified=False,
                hardware_flashed=False,command=args)
    (OUT/'VALIDACION_CUBEIDE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if code or len(elfs)!=2:raise SystemExit(2)


if __name__=='__main__':main()
