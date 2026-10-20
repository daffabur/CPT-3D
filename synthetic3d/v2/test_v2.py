import unittest
import numpy as np
from .core import Grid, generate, extract_cpt, validate, interpret, classify_sbt, assign_split, validate_manifest

class V2Tests(unittest.TestCase):
    def make(self, family='horizontal',shape=(16,16,8),seed=1):
        return generate(Grid(*shape,2,2,1),family,2,seed,scenario_id=family,parameter_family_id='standard',realization_id=f'{family}-{seed}')
    def test_reproducibility_and_separate_targets(self):
        a=self.make(); b=self.make()
        for key in ('unit_id','qc_mpa','fs_mpa','Ic','sbt_id'):
            self.assertTrue(np.array_equal(a[key],b[key]))
        self.assertEqual(validate(a),[])
        self.assertFalse(np.array_equal(a['unit_id'],a['sbt_id']))
    def test_shapes_and_coordinates(self):
        for shape in ((16,16,8),(12,10,12)):
            a=self.make(shape=shape)
            g=Grid(*shape,2,2,1,100,200,25,50)
            self.assertEqual(g.index_xy(*g.xy_m(3,4)),(3,4))
            self.assertEqual(a['Ic'].shape,shape)
    def test_cpt_and_mask(self):
        a=self.make(); one=extract_cpt(a,[(3,4)]); many=extract_cpt(a,[(3,4),(7,8)])
        self.assertEqual(validate(a,one),[])
        self.assertEqual(validate(a,many),[])
        self.assertEqual(int(one['observed_mask'].sum()),8)
        self.assertEqual(int(many['observed_mask'].sum()),16)
    def test_formula_domain_and_sbt(self):
        a=self.make(); self.assertTrue(np.array_equal(classify_sbt(a['Ic']),a['sbt_id']))
        with self.assertRaisesRegex(ValueError,'explicit'):
            interpret(a['qc_mpa'],a['fs_mpa'],a['sigma_v0_mpa'],a['sigma_v0_eff_mpa'])
        bad=dict(a); bad['Ic']=a['Ic'].copy(); bad['Ic'][0,0,0]+=1
        self.assertIn('Ic: formula mismatch',validate(bad))
        bad=dict(a); bad['qc_mpa']=a['qc_mpa'].copy(); bad['qc_mpa'][0,0,0]=0
        self.assertTrue(any('CPT domain' in message for message in validate(bad)))
        bad=dict(a); bad['Ic']=a['Ic'].copy(); bad['Ic'][0,0,0]=np.nan
        self.assertTrue(any('Ic: NaN/Inf' in message for message in validate(bad)))
    def test_split_leakage(self):
        a=self.make('lens',shape=(16,16,12)); self.assertEqual(assign_split(a['metadata']),'test_family')
        records=[{'realization_id':'x','split':'train','family':'horizontal','parameter_family_id':'standard'},
                 {'realization_id':'x','split':'test_id','family':'horizontal','parameter_family_id':'standard'}]
        self.assertTrue(validate_manifest(records))

if __name__=='__main__': unittest.main()
