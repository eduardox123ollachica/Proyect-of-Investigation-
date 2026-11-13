"""Prepara corrección mínima y copia de prueba IDE; no cambia D: ni entrega v1."""
from pathlib import Path
import shutil
import xml.dom.minidom as DOM
import xml.etree.ElementTree as ET
import json
import hashlib

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/correccion_cubeide_2026-10-22'
SOURCE=Path(r'D:\STM32IDE\Prueba_2_CWRU_Extraccion')
NAME='Prueba_2_CWRU_Extraccion'


def main():
    if OUT.exists():raise FileExistsError('No sobrescribir una revisión de corrección')
    OUT.mkdir(parents=True)
    original=(SOURCE/'.cproject').read_bytes()
    if b'<?fileVersion' in original:raise ValueError('Ya contiene marcador; revisar otra causa')
    (OUT/'cproject_original_sin_marcador.xml').write_bytes(original)
    close=original.index(b'?>')+2
    corrected=original[:close]+b'\n<?fileVersion 4.0.0?>'+original[close:]
    (OUT/'.cproject').write_bytes(corrected)
    # La única diferencia admitida es la inserción de la instrucción requerida.
    assert corrected.replace(b'\n<?fileVersion 4.0.0?>',b'',1)==original
    doc=DOM.parseString(corrected)
    assert doc.firstChild.nodeType==doc.PROCESSING_INSTRUCTION_NODE and doc.firstChild.target=='fileVersion'
    cfg=ET.fromstring(corrected).findall('./storageModule/cconfiguration')
    assert sorted(c.find('storageModule').get('name') for c in cfg)==['Debug','Release']
    project=OUT/'prueba_ide'/NAME
    project.mkdir(parents=True)
    # Sólo recursos necesarios para el build: no copia archivos de salida antiguos.
    dirs=['Inc','Src','Startup','.settings','Drivers/CMSIS/Include',
          'Drivers/CMSIS/Device/ST/STM32F4xx/Include','Drivers/STM32F4xx_HAL_Driver/Inc',
          'Middlewares/CMSIS-DSP/Include','Middlewares/CMSIS-DSP/PrivateInclude',
          'Middlewares/CMSIS-DSP/Source/TransformFunctions','Middlewares/CMSIS-DSP/Source/ComplexMathFunctions',
          'Middlewares/CMSIS-DSP/Source/CommonTables']
    for relative in dirs:shutil.copytree(SOURCE/relative,project/relative)
    for relative in ('.project','STM32F446RETX_FLASH.ld','STM32F446RETX_RAM.ld'):
        shutil.copy2(SOURCE/relative,project/relative)
    (project/'.cproject').write_bytes(corrected)
    report=dict(issue='Missing CDT fileVersion processing instruction',
                source=str(SOURCE),original_sha256=hashlib.sha256(original).hexdigest(),
                corrected_sha256=hashlib.sha256(corrected).hexdigest(),
                change='Insert <?fileVersion 4.0.0?> after XML declaration; no other bytes changed',
                configurations=['Debug','Release'],original_delivery_and_d_unchanged=True,
                primary_source='https://github.com/eclipse-cdt/cdt/blob/main/core/org.eclipse.cdt.core/model/org/eclipse/cdt/internal/core/settings/model/xml/XmlProjectDescriptionStorage.java')
    (OUT/'DIAGNOSTICO.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
