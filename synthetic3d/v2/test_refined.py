import copy
import unittest
import numpy as np
from .core import Grid, interpret, classify_sbt, extract_cpt
from .refined import (default_geometry, default_response, generate, validate,
                      validate_reproducibility, split, validate_manifest, response_diagnostics)

class RefinedTests(unittest.TestCase):
    def make(self,family='undulating',n=3,seed=9,geometry=None,response=None):
        return generate(Grid(16,16,16,2,2,1),family,n,seed,scenario_id=family,
                        parameter_family_id='standard',realization_id=f'{family}-{seed}',
                        geometry=geometry,response=response)

    def test_independent_numeric_ic_and_cptu(self):
        # Independently calculated: (10-.1)/.08=123.75; 100*.1/9.9=1.010101...;
        # sqrt((3.47-log10(123.75))^2+(log10(1.010101...)+1.22)^2)=1.8429462503823768.
        qc=np.array([10.],dtype=np.float32); fs=np.array([.1],dtype=np.float32)
        total=np.array([.1],dtype=np.float32); effective=np.array([.08],dtype=np.float32)
        qt,Qt,Fr,Ic=interpret(qc,fs,total,effective,qt_approximation=True)
        self.assertAlmostEqual(float(qt[0]),10.,places=6)
        self.assertAlmostEqual(float(Qt[0]),123.75,places=4)
        self.assertAlmostEqual(float(Fr[0]),1.0101010101,places=5)
        self.assertAlmostEqual(float(Ic[0]),1.84294625038,places=5)
        corrected,*_=interpret(qc,fs,total,effective,u2_mpa=np.array([.2],dtype=np.float32),area_ratio=.8)
        self.assertAlmostEqual(float(corrected[0]),10.04,places=5)

    def test_sbt_boundary_table_and_invalid(self):
        values=np.array([.99,1.,1.3099,1.31,2.0499,2.05,2.5999,2.60,2.9499,2.95,3.5999,3.60,4.,4.01])
        self.assertEqual(classify_sbt(values).tolist(),[0,7,7,6,6,5,5,4,4,3,3,2,2,0])
        with self.assertRaises(ValueError): classify_sbt(np.array([np.nan]))
        with self.assertRaisesRegex(ValueError,'explicit'):
            interpret(np.array([1.]),np.array([.01]),np.array([.1]),np.array([.08]))
        with self.assertRaisesRegex(ValueError,'invalid domain'):
            interpret(np.array([.05]),np.array([.01]),np.array([.1]),np.array([.08]),qt_approximation=True)

    def test_boundary_variation_and_rejection(self):
        a=self.make(); self.assertEqual(validate(a),[])
        thickness=np.diff(a['boundaries_m'],axis=0)
        self.assertGreater(float(np.ptp(thickness[0])),.02)
        g=default_geometry(3,'undulating'); g['boundary_wave_fraction'][0]=.025
        b=self.make(geometry=g)
        self.assertFalse(np.array_equal(a['boundaries_m'],b['boundaries_m']))
        g['boundary_wave_fraction'][0]=.5
        with self.assertRaisesRegex(ValueError,'boundary crossing'):
            self.make(geometry=g)
        common=default_geometry(3,'undulating',mode='common_shift')
        self.assertEqual(validate(self.make(geometry=common)),[])

    def test_lens_geometry_and_parameter_change(self):
        g=default_geometry(4,'lens')
        a=self.make('lens',4,geometry=g)
        self.assertEqual(validate(a),[])
        self.assertEqual(validate_reproducibility(a),[])
        g2=copy.deepcopy(g);g2['lens']['center_x_fraction']=.44;g2['lens']['radius_x_fraction_domain']=.16
        b=self.make('lens',4,geometry=g2)
        self.assertNotEqual(a['metadata']['lens']['center_local_m'],b['metadata']['lens']['center_local_m'])
        self.assertFalse(np.array_equal(a['unit_id'],b['unit_id']))
        self.assertGreater(a['metadata']['lens']['voxel_count'],7)
        g2['lens']['radius_x_fraction_domain']=.8
        with self.assertRaisesRegex(ValueError,'lens exits'):
            self.make('lens',4,geometry=g2)

    def test_response_change_overlap_and_mask(self):
        a=self.make('horizontal',4)
        response=default_response(4,9); response['units'][0]['qc_mpa_median']*=1.2
        b=self.make('horizontal',4,response=response)
        self.assertTrue(np.array_equal(a['unit_id'],b['unit_id']))
        self.assertFalse(np.array_equal(a['qc_mpa'],b['qc_mpa']))
        self.assertGreater(response_diagnostics(a)['qc_p10_p90_overlap_pairs'],0)
        one=extract_cpt(a,[(2,3)]); many=extract_cpt(a,[(2,3),(7,8)])
        self.assertEqual(validate(a,one),[]);self.assertEqual(validate(a,many),[])
        self.assertEqual(int(one['observed_mask'].sum()),16)
        response['low_pore_correction_assumed']=False
        with self.assertRaisesRegex(ValueError,'unsupported'):
            self.make('horizontal',4,response=response)

    def test_seed_and_split(self):
        a=self.make();b=self.make();self.assertTrue(np.array_equal(a['Ic'],b['Ic']))
        self.assertEqual(validate_reproducibility(a),[])
        with self.assertRaisesRegex(ValueError,'spacings positive'):
            Grid(16,16,8,float('nan'),2,1)
        lens=self.make('lens',4);self.assertEqual(split(lens['metadata']),'test_family')
        e={'realization_id':'x','realization_seed':1,'scenario_id':'horizontal','parameter_family_id':'standard','family':'horizontal','split':'train'}
        self.assertTrue(validate_manifest([e,e]))

if __name__=='__main__': unittest.main()
