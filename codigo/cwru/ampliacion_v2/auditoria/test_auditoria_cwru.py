"""Pruebas sintéticas y negativas. Nunca modifican el dataset original."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import upfirdn
from scipy.stats import kurtosis,skew

from audit_core import (affine_relation,assert_metadata,assert_windows,block_fingerprints,choose_channel,
                        file_hash,independent_decimation,independent_features,numeric_hash,
                        scan_exact_blocks,verify_manifest)


class AuditTests(unittest.TestCase):
    def test_explicit_channel_rejects_auxiliary_key(self):
        correct = np.arange(20,dtype=float)
        d = {"X217_DE_time":correct+100,"X175_DE_time":correct}
        np.testing.assert_array_equal(choose_channel(d,"X175_DE_time",20),correct)
        with self.assertRaises(ValueError):choose_channel(d,"X175_DE_time",21)
        with self.assertRaises(ValueError):choose_channel(d,"X999_DE_time",20)

    def test_fingerprints_all_offsets_and_signed_zero(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(80,2));b = rng.normal(size=(90,2))
        b[13:44] = a[17:48]
        h1,h2 = block_fingerprints(a),block_fingerprints(b)
        self.assertEqual(h1[17],h2[13])
        records = [{"file":"a","split":"train"},{"file":"b","split":"test"}]
        matches,summary = scan_exact_blocks(records,[a,b])
        self.assertTrue(summary["coverage_complete"])
        self.assertTrue(any(r['start_a']==17 and r['start_b']==13 and r['length_samples']==31 for r in matches))
        self.assertEqual(numeric_hash(np.array([-0.0])),numeric_hash(np.array([0.0])))

    def test_affine_copy_detected_but_unrelated_signal_rejected(self):
        rng=np.random.default_rng(9);a=rng.normal(size=300)
        self.assertTrue(affine_relation(a,3*a+0.25)["affine_match"])
        self.assertFalse(affine_relation(a,rng.normal(size=300))["affine_match"])

    def test_independent_fir_matches_polyphase_reference(self):
        rng=np.random.default_rng(11);signal=rng.normal(size=8195)
        taps=np.hanning(177).astype(np.float32);taps/=taps.sum()
        expected=upfirdn(taps,signal,down=4)
        expected=expected[22:22+(len(signal)+3)//4][22:-22].astype(np.float32)
        np.testing.assert_allclose(independent_decimation(signal,taps),expected,rtol=1e-6,atol=1e-7)

    def test_independent_feature_moments_and_spectral_bins(self):
        rng=np.random.default_rng(22);w=rng.normal(size=(4096,2))
        w[:,0]+=0.4*np.sin(2*np.pi*123*np.arange(4096)/4096)
        f=independent_features(w)
        for j in range(2):
            self.assertAlmostEqual(f[j*7+1],skew(w[:,j],bias=False),places=10)
            self.assertAlmostEqual(f[j*7+2],kurtosis(w[:,j],fisher=True,bias=False),places=10)
            self.assertAlmostEqual(f[j*7+4]/(12000/4096),round(f[j*7+4]/(12000/4096)),places=10)
            self.assertTrue(0<=f[j*7+6]<=1)

    def test_constant_signal_has_finite_features(self):
        f=independent_features(np.ones((4096,2)))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[0],1.0);self.assertEqual(f[1],0.0);self.assertEqual(f[2],0.0)

    def test_reversed_channels_and_wrong_window_offset_are_detected(self):
        rng=np.random.default_rng(33);a=rng.normal(size=(5000,2)).astype(np.float32)
        assert_windows(a,a[123:4219],123)
        with self.assertRaises(ValueError):assert_windows(a,a[123:4219,::-1],123)
        with self.assertRaises(ValueError):assert_windows(a,a[123:4219],124)

    def test_output_manifest_detects_tampering(self):
        # Usa el directorio de ejecución autorizado, no un TEMP del sistema.
        with tempfile.TemporaryDirectory(prefix="cwru_audit_test_", dir=Path.cwd()) as tmp:
            p=Path(tmp); data=p/"fixture.txt"
            data.write_text("original",encoding="utf-8")
            with (p/"MANIFEST_SHA256.csv").open("w",newline="",encoding="utf-8-sig") as f:
                writer=csv.DictWriter(f,["path","sha256"]);writer.writeheader()
                writer.writerow({"path":data.name,"sha256":file_hash(data)})
            self.assertEqual(len(verify_manifest(p)),1)
            data.write_text("tampered",encoding="utf-8")
            with self.assertRaises(ValueError):verify_manifest(p)

    def test_metadata_decimal_rounding_not_real_timestamp_or_key_change(self):
        base=dict(start_time_s='0.0018333333333333333',de_key='X175_DE_time')
        assert_metadata(base,dict(start_time_s='0.0018333333333333',de_key='X175_DE_time'))
        with self.assertRaises(ValueError):
            assert_metadata(base,dict(start_time_s='0.1018333333333333',de_key='X175_DE_time'))
        with self.assertRaises(ValueError):
            assert_metadata(base,dict(start_time_s=base['start_time_s'],de_key='X217_DE_time'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
