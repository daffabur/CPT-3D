import copy
import unittest
import numpy as np
from .core import Grid, extract_cpt
from .geotechnical_v3 import DEFAULT_RESPONSE, generate, validate, direction_autocorrelation

class GeotechnicalV3Tests(unittest.TestCase):
    def make(self,seed=1,**kw):
        return generate(Grid(12,12,12,2,2,1),'horizontal',3,seed,scenario_id='test',realization_id='test',**kw)
    def test_physical_equations_independent(self):
        v=self.make();g=v['gamma_kn_m3'].astype(float);s=v['sigma_v0_mpa'].astype(float)
        self.assertTrue(np.allclose(s[:,:,0],g[:,:,0]/2000,rtol=1e-6))
        self.assertTrue(np.allclose(s[:,:,1],(g[:,:,0]+g[:,:,1]/2)/1000,rtol=1e-6))
        self.assertTrue(np.array_equal(s,v['sigma_v0_eff_mpa']))
        self.assertTrue(np.array_equal(v['u0_mpa'],np.zeros_like(v['u0_mpa'])))
        self.assertTrue(np.allclose(v['Qt'],(v['qt_mpa']-s)/s,rtol=1e-5))
        self.assertTrue(np.allclose(v['Fr_percent'],100*v['fs_mpa']/(v['qt_mpa']-s),rtol=1e-5))
        self.assertEqual(validate(v,extract_cpt(v,[(1,1)])),[])
    def test_mixture_material_and_reproducibility(self):
        a=self.make();b=self.make();self.assertTrue(np.array_equal(a['Ic'],b['Ic']))
        c=self.make(material_seed=99)
        self.assertTrue(np.array_equal(a['unit_id'],c['unit_id']))
        self.assertFalse(np.array_equal(a['material_group'],c['material_group']))
        self.assertFalse(np.array_equal(a['Ic'],c['Ic']))
        self.assertGreater(len(np.unique(a['material_group'])),1)
        self.assertGreater(len(np.unique(a['sbt_id'])),1)
    def test_covariance_rejected(self):
        r=copy.deepcopy(DEFAULT_RESPONSE);r['log_covariance']=[[1,2],[2,1]]
        with self.assertRaisesRegex(ValueError,'positive definite'):self.make(response=r)
    def test_directional_formula(self):
        a=np.arange(64,dtype=float).reshape(4,4,4)
        for axis in range(3):
            self.assertAlmostEqual(direction_autocorrelation(a,axis),1.0)
        with self.assertRaises(ValueError):direction_autocorrelation(a,3)
    def test_observation_minimum(self):
        v=self.make();self.assertEqual(validate(v,extract_cpt(v,[(1,1)])),[])
        bad={'profiles':[],'observed_mask':np.zeros(v['Ic'].shape,bool)}
        self.assertIn('observation missing',validate(v,bad))
if __name__=='__main__':unittest.main()
