"""Migración mecánica final de opciones CubeIDE en la copia preparada."""
from pathlib import Path
import xml.etree.ElementTree as E
import copy

ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'despliegue_stm32/cwru_extraccion_prueba2/Prueba_2_CWRU_Extraccion/.cproject'
t=E.parse(P)
original=E.parse(Path(r'D:\STM32IDE\Prueba 2\.cproject'))
reference=next(c for c in original.findall('./storageModule/cconfiguration') if '.debug.' in c.get('id'))
for cfg in t.findall('./storageModule/cconfiguration'):
    name=cfg.find('storageModule').get('name')
    for suffix in ('tool.c.compiler','tool.cpp.compiler'):
        ref=next(e for e in reference.findall('.//tool') if e.get('superClass','').endswith(suffix))
        tool=next(e for e in cfg.findall('.//tool') if e.get('superClass','').endswith(suffix))
        for opt_suffix in ('includepaths','definedsymbols'):
            source=next(e for e in ref.findall('option') if e.get('superClass','').endswith('.'+opt_suffix))
            target=next(e for e in tool.findall('option') if e.get('superClass','').endswith('.'+opt_suffix))
            for child in list(target):target.remove(child)
            for child in source:target.append(copy.deepcopy(child))
    for e in cfg.findall('.//option'):
        if e.get('id','').endswith('.cwru'):e.set('id',e.get('id')+'.'+name)
        if e.get('superClass','').endswith('tool.c.linker.option.systemcalls'):
            e.set('value','com.st.stm32cube.ide.mcu.gnu.managedbuild.tool.c.linker.option.systemcalls.value.minimalimplementation')
        if name=='Release' and e.get('valueType')=='definedSymbols':
            for child in list(e):
                if child.get('value')=='DEBUG':e.remove(child)
    for tool in cfg.findall('.//tool'):
        if not tool.get('superClass','').endswith(('tool.c.linker','tool.cpp.linker')):continue
        prefix=tool.get('superClass')+'.option.otherflags'
        options=[o for o in tool.findall('option') if o.get('superClass')==prefix]
        opt=options[0] if options else E.SubElement(tool,'option',{'id':prefix+'.cwru.'+name,'superClass':prefix,'valueType':'stringList','name':'Other flags'})
        if not any('_printf_float' in c.get('value','') for c in opt):
            E.SubElement(opt,'listOptionValue',{'builtIn':'false','value':'-u _printf_float'})
E.indent(t,space='\t')
P.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<?fileVersion 4.0.0?>\n'+E.tostring(t.getroot(),encoding='unicode')+'\n',encoding='utf-8')
print('Configuraciones Debug/Release actualizadas sólo en la copia')
