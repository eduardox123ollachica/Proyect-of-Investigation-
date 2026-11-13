"""Pruebas de solo lectura: configuracion16→17 y dominio numerico del override."""
import unittest
import xml.etree.ElementTree as ET
import preparar_total_single_rounding_v2 as m


class SingleRoundingPreparationTests(unittest.TestCase):
    def test_preserves_sixteen_configurations_and_processing_instruction(self):
        p=m.prior();root,settings,old=p.configs(m.SOURCE/'.cproject')
        snapshot={k:ET.tostring(v) for k,v in old.items()}
        new=m.add_configuration(p,root,settings,old)
        self.assertEqual(len(settings),17)
        self.assertTrue(all(ET.tostring(v)==snapshot[k] for k,v in old.items()))
        oldids={n.get('id') for c in old.values() for n in c.iter() if n.get('id')}
        self.assertFalse(oldids.intersection(n.get('id') for n in new.iter() if n.get('id')))
        self.assertIn(b'<?fileVersion 4.0.0?>',p.xml_bytes(root))

    def test_two_compilers_receive_single_macros_and_only_one_source_entry(self):
        p=m.prior();root,settings,old=p.configs(m.SOURCE/'.cproject')
        new=m.add_configuration(p,root,settings,old)
        for tool in new.iter('tool'):
            if tool.get('superClass','').endswith(('tool.c.compiler','tool.cpp.compiler')):
                values=[n.get('value') for n in p.option(tool,'.definedsymbols')]
                self.assertEqual(values.count('CMSIS_NN_USE_SINGLE_ROUNDING=1'),1)
                self.assertEqual(values.count('TFLITE_SINGLE_ROUNDING=1'),1)
                self.assertIn('../TFLM/third_party/cmsis_nn/Include',
                              [n.get('value') for n in p.option(tool,'.includepaths')])
        before=[n.get('name') for n in old[m.BASE].find('.//sourceEntries')]
        after=[n.get('name') for n in new.find('.//sourceEntries')]
        self.assertEqual(after,before+['CWRU_TotalSR'])

    def test_rejects_unexpected_original_configuration_count(self):
        p=m.prior();root,settings,old=p.configs(m.SOURCE/'.cproject');old.pop(m.ENTM)
        with self.assertRaises(RuntimeError):m.add_configuration(p,root,settings,old)

    def test_archive_member_defines_only_the_overridden_global(self):
        body=m.archive_member((m.SOURCE/m.LIBRARY).read_bytes(),m.FUNC+'.o')
        self.assertEqual(m.global_definitions(body),[m.FUNC])
        self.assertEqual(m.sha(m.SOURCE/m.LIBRARY),m.LIBRARY_SHA)

    def test_malformed_archive_cannot_be_used(self):
        with self.assertRaises(RuntimeError):m.archive_member(b'bad',m.FUNC+'.o')

    def test_all_models_use_supported_path_and_shift_ranges(self):
        scope=m.parameter_scope()
        self.assertEqual(scope['channels'],1056)
        self.assertEqual(len(scope['models']),32)
        self.assertTrue(all(r['all_FC_per_channel']
            and -31<=r['minimum_shift']<=r['maximum_shift']<30
            and 0<=r['softmax_multiplier_shift']<30 for r in scope['models']))

    def test_closed_diagnosis_has_expected_manifest_and_causal_checks(self):
        manifest,digest=m.diagnosis_check(m.DIAGNOSIS)
        self.assertEqual(digest,'8be8a8ec59aec4ad4fac58a092e7dc64f1c5a5a3e66339ab430801bf09aca73c')
        self.assertEqual(manifest.name,'MANIFEST_DIAGNOSTICO_ROUNDING_SHA256.json')

    def test_link_recipe_resolves_application_objects_before_archive(self):
        folder=m.OLD/'copia_aislada'/m.NAME/m.BASE
        recipe=(folder/'makefile').read_text(encoding='utf-8')
        libraries=(folder/'objects.mk').read_text(encoding='utf-8')
        self.assertTrue(m.link_order(recipe,libraries)['application_objects_before_prebuilt_archive'])
        swapped=recipe.replace('@"objects.list" $(USER_OBJS) $(LIBS)','$(LIBS) $(USER_OBJS) @"objects.list"')
        self.assertFalse(m.link_order(swapped,libraries)['application_objects_before_prebuilt_archive'])

    def test_definition_report_distinguishes_absence_from_single_rounding(self):
        old=m.single_definition_report(['gcc -O3 -c example.c'],m.BASE)
        new=m.single_definition_report(['gcc -DCMSIS_NN_USE_SINGLE_ROUNDING=1 -DTFLITE_SINGLE_ROUNDING=1 -c example.c'],m.CONFIG)
        for key in ('CMSIS_NN_USE_SINGLE_ROUNDING','TFLITE_SINGLE_ROUNDING'):
            self.assertTrue(old[key+'_matches_expected']);self.assertEqual(old['compile_definition_values'][key],[[]])
            self.assertTrue(new[key+'_matches_expected']);self.assertEqual(new['compile_definition_values'][key],[['1']])
        missing=m.single_definition_report(['gcc -c example.c'],m.CONFIG)
        self.assertFalse(missing['TFLITE_SINGLE_ROUNDING_matches_expected'])
        self.assertFalse(m.single_definition_report([],m.CONFIG)['CMSIS_NN_USE_SINGLE_ROUNDING_matches_expected'])


if __name__=='__main__':unittest.main(verbosity=2)
