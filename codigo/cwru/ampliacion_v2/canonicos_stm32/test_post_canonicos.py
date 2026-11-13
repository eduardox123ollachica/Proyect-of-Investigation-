"""Contrato estático POST y compilación ARM de los .inc reales, sin placa.

No sustituye el gate UART ni ejecuta aritmética C: la compilación usa mocks
para los periféricos y las tres etapas ya existentes. Comprueba tipos,
formato, cabecera C/C++, límites del mensaje y publicación de resultados.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LEGACY=ROOT/'codigo/cwru/ampliacion_v2/costo_total'


def load(name):
    spec=importlib.util.spec_from_file_location('post_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def text(name):return (HERE/name).read_text(encoding='utf-8')


def result_fields(source,prefix):
    match=re.search(r'"('+re.escape(prefix)+r',[^"\n]+)"',source)
    if match is None:raise AssertionError('Formato de respuesta ausente: '+prefix)
    return match.group(1).removesuffix(r'\r\n').split(',')


class PostContract(unittest.TestCase):
    def test_response_layout_preserves_raw_field_positions(self):
        raw=result_fields((LEGACY/'cwru_total_commands.inc').read_text(encoding='utf-8'),'CWRUTOTAL_RESULT')
        post=result_fields(text('cwru_canon_post_commands.inc'),'CWRUTOTAL_POST_RESULT')
        self.assertEqual(len(raw),len(post))
        self.assertEqual(len(post),19)
        # Prefix distinguishes POST. The remaining slots retain RAW meanings.
        for index in (1,2,3,5,11,12,13,14,15,16,17):
            self.assertEqual(raw[index],post[index],index)
        self.assertEqual([post[i] for i in (4,6,7,8,9,10,18)],['0']*7)
        self.assertEqual(post[17],'%lu')

    def test_route_and_uart_budget_are_compatible(self):
        source=text('cwru_canon_post_commands.inc')
        self.assertIn('const char *prefix="CWRUTOTAL_POST,";',source)
        self.assertIn('"CWRUTOTAL_POST_INFO,1,14,56,0\\r\\n"',source)
        # 14 binary32 words -> 56 bytes -> 112 hex chars plus checksum/comma.
        payload='00000000,'+'00'*56
        message='CWRUTOTAL_POST,'+payload
        self.assertEqual(len(payload),121)
        self.assertEqual(len(message),136)
        self.assertLess(len(message)+2,1024)
        self.assertIn("strlen(p)!=121U || p[8]!=','",source)
        self.assertIn('unsigned char bytes[56]',source)
        self.assertIn('float features[14];memcpy(features,bytes,sizeof(features));',source)
        # Legacy command router actually forwards the chosen namespace.
        router=(Path('D:/STM32IDE/Prueba_2_CWRU_Extraccion/Src/uart_commands.c')).read_text(encoding='utf-8-sig')
        self.assertRegex(router,r'strncmp\s*\([^;\n]*"CWRUTOTAL_"')

    def test_failed_requests_cannot_publish_a_new_result(self):
        source=text('cwru_canon_post_commands.inc')
        body=source[source.index('CWRUTOTAL_Invalidate();\n    const char *p='):]
        execute=body.index('CWRUCANON_PostExecute(')
        # All transport/runtime guards precede the shared-stage call.
        for reason in ('FORMAT','CHECKSUM','NONFINITE','DWT_DISABLED','FPSCR','NO_SELECTION'):
            self.assertLess(body.index('"'+reason+'"'),execute,reason)
        self.assertGreater(body.index('"EXECUTE"'),execute)
        self.assertIn('uint32_t next=g_total_id+1U;if(next==0U) next=1U;',body)
        self.assertRegex(body,r'if\(formatted\)\s*\{[^}]*g_total_id=next;[^}]*g_total_checksum=checksum;[^}]*g_total_valid=1U;')
        self.assertEqual(body.count('g_total_valid=1U'),1)

    def test_post_uses_shared_stages_without_raw_extraction(self):
        source=text('cwru_canon_post_math.inc')
        self.assertNotIn('CWRUTOTAL_Execute(',source)
        self.assertNotIn('CWRUMASK_Extract(',source)
        self.assertNotIn('g_window',source)
        self.assertIn('memset(out,0,sizeof(*out));',source)
        self.assertIn('out->features[d->indices[j]]=features[d->indices[j]];',source)
        positions=[source.index(call) for call in ('input_transform(d,out)','CWRUTNN_Predict(','normality(d,out)')]
        self.assertEqual(positions,sorted(positions))
        self.assertNotRegex(source,r'out->extractions\s*=(?!=)')
        self.assertLess(source.index('isfinite(features[i])'),source.index('memset(out'))
        for stage in (2,3,4):self.assertIn('out->failed_stage='+str(stage)+'U;',source)


HARNESS=r'''
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cwru_canon_post.h"
#include "cwru_total_nn.h"
typedef struct { volatile uint32_t CTRL,CYCCNT; } TestDWT;
extern TestDWT test_dwt;
#define DWT (&test_dwt)
#define DWT_CTRL_CYCCNTENA_Msk 1U
extern uint32_t __get_FPSCR(void);
extern int total_format(char *,size_t,const char *,...) __attribute__((format(printf,3,4)));
extern void total_error(char *,size_t,const char *,long);
extern int total_checksum(const char *,uint32_t *);
extern uint32_t total_bits(float);
static CWRUTOTAL_Result g_total_result;
static uint32_t g_total_id,g_total_checksum,g_total_representative;
static uint8_t g_total_valid;
static int input_transform(const CWRUTV_Descriptor *d,CWRUTOTAL_Result *r) { (void)d;(void)r;return 0; }
static int normality(const CWRUTV_Descriptor *d,CWRUTOTAL_Result *r) { (void)d;(void)r;return 0; }
#include "cwru_canon_post_math.inc"
#include "cwru_canon_post_commands.inc"
int test_route(char *command,char *out,size_t cap) { return CWRUCANON_Command(command,out,cap); }
_Static_assert(sizeof(float)==4,"POST requires binary32 storage");
_Static_assert(sizeof(((CWRUTOTAL_Result*)0)->features)==56,"POST payload width");
'''


class PostARMCompilation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare=load('preparar_proyecto');cls.export=load('exportar_catalogo')
        tools,_=cls.prepare.toolchain()
        cls.cc=tools/'arm-none-eabi-gcc.exe'
        cls.base=(ROOT/'tmp').resolve();cls.base.mkdir(exist_ok=True)
        cls.folder=Path(tempfile.mkdtemp(prefix='post_static_',dir=cls.base)).resolve()
        if not cls.folder.is_relative_to(cls.base):raise RuntimeError('Fixture fuera del workspace')
        for name in ('cwru_total.h','cwru_total_nn.h'):
            shutil.copy2(LEGACY/name,cls.folder/name)
        (cls.folder/'cwru_total_catalog.h').write_text(cls.export.render_header(),encoding='utf-8')
        (cls.folder/'cwru_masks.h').write_text('''#include <stdint.h>
typedef float float32_t;
typedef struct { int unused; } CWRU_FeatureExtractor;
#define CWRU_WINDOW_SAMPLES 4096
#define CWRU_NUM_CHANNELS 2
#define CWRU_FE_SCRATCH_FLOATS 4096
''',encoding='utf-8')
        (cls.folder/'post.c').write_text(HARNESS,encoding='utf-8')
        (cls.folder/'header.cpp').write_text('#include "cwru_canon_post.h"\nstatic_assert(sizeof(float)==4,"binary32");\n',encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        if cls.folder!=cls.base and cls.folder.is_relative_to(cls.base):shutil.rmtree(cls.folder)

    def compile(self,filename,language):
        args=[str(self.cc),'-mcpu=cortex-m4','-mthumb','-mfpu=fpv4-sp-d16','-mfloat-abi=hard',
            '-x',language,'-std=gnu11' if language=='c' else '-std=gnu++17',
            '-fsyntax-only','-Wall','-Wextra','-Werror','-Wformat=2',
            '-I'+str(self.folder),'-I'+str(HERE),str(self.folder/filename)]
        result=subprocess.run(args,capture_output=True,text=True,timeout=30,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_real_post_templates_compile_for_ARM_with_format_checks(self):self.compile('post.c','c')
    def test_public_header_is_compatible_with_cpp(self):self.compile('header.cpp','c++')


if __name__=='__main__':unittest.main()
