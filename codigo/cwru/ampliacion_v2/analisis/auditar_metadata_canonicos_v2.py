"""Comprueba el único metadato CDT regenerado; conserva sellos anteriores."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
PREPARER = ROOT/'codigo/cwru/ampliacion_v2/canonicos_stm32/preparar_proyecto.py'


def load_preparer():
    spec = importlib.util.spec_from_file_location('canon_metadata_preparer', PREPARER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(node, configuration_identity=True):
    ignored = {'env-hash'}
    if not configuration_identity and node.tag == 'configuration':
        ignored |= {'id', 'name'}
    return (node.tag, tuple(sorted((k, v) for k, v in node.attrib.items() if k not in ignored)),
        (node.text or '').strip(), tuple(canonical(child, configuration_identity) for child in node))


def audit():
    p = load_preparer()
    before_path = p.SOURCE/'.settings/language.settings.xml'
    after_path = p.PROJECT/'.settings/language.settings.xml'
    prep = p.read(p.PREPARATION)
    relative = '.settings/language.settings.xml'
    p.require(p.sha(before_path) == prep['D_source_sha256_before'][relative],
        'El metadato de origen difiere de la instantánea anterior')
    before_root, after_root = ET.parse(before_path).getroot(), ET.parse(after_path).getroot()
    p.require(before_root.tag == after_root.tag and before_root.attrib == after_root.attrib,
        'Contenedor de metadatos diferente')
    before = {node.get('id'): node for node in before_root}
    after = {node.get('id'): node for node in after_root}
    p.require(len(before_root) == len(before) == 17 and len(after_root) == len(after) == 18,
        'Número o unicidad de configuraciones de lenguaje diferente')
    p.require(all(key in after and canonical(node) == canonical(after[key]) for key, node in before.items()),
        'El IDE modificó semánticamente una configuración anterior')
    added = [node for key, node in after.items() if key not in before]
    base = next(node for node in before.values() if node.get('name') == p.BASE)
    p.require(len(added) == 1 and added[0].get('name') == p.CONFIG,
        'No se añadió exclusivamente la configuración canónica')
    p.require(canonical(base, False) == canonical(added[0], False),
        'Los providers canónicos no coinciden con la plantilla SR')
    config = p.helper().configs(p.PROJECT/'.cproject')[2][p.CONFIG]
    p.require(config.get('id') == added[0].get('id'), 'ID de lenguaje no coincide con CDT')
    return dict(status='PASS_GENERATED_METADATA_ONLY', before_sha256=p.sha(before_path),
        after_sha256=p.sha(after_path), original_configurations_semantically_equal=17,
        ignored_attribute='env-hash', added_configuration=p.CONFIG,
        added_providers_equal_SR_except_identity_and_env_hash=True,
        original_manifest_preserved=True, verifier_sha256=p.sha(__file__),
        D_files_modified=False, serial_opened=False, hardware_flashed=False)


if __name__ == '__main__':
    p = load_preparer()
    result = audit()
    p.write(p.OUT/'AUDITORIA_METADATA_CDT.json', result)
    print(json.dumps(result, ensure_ascii=True))
