"""Tests for frozen pilot planning, file integrity and crash-resume recovery."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from collections import Counter
import numpy as np
from .v3_observation_policy import variants,policy
from .v3_dataset_production import grid_for,_read_saved
from .v3_dataset_production import (CONFIG_PATH,build_plan,config_checks,digest,initialize,
                                    produce,validate_dataset,validate_entry)

class DatasetProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(CONFIG_PATH.read_text())
        cls.root=Path(cls.config['output_root'])
        cls.manifest=json.loads((cls.root/'manifest.json').read_text())

    def test_pilot_design_and_lineage(self):
        plan=build_plan(self.config,'pilot')
        self.assertEqual(len(plan['entries']),45)
        self.assertEqual(plan['config_sha256'],digest(self.config))
        self.assertEqual(Counter(e['split'] for e in plan['entries']),self.config['pilot_counts'])
        self.assertEqual(len({e['lineage_group'] for e in plan['entries']}),45)
        self.assertEqual(len({(e['geology_seed'],e['material_seed'],e['site_seed']) for e in plan['entries']}),45)
        self.assertEqual(set(e['n_cpt'] for e in plan['entries']),set(self.config['stored_cpt_counts']))
        for split in self.config['pilot_counts']:
            self.assertEqual(Counter(e['grid_name'] for e in plan['entries'] if e['split']==split),
                             {g['name']:self.config['pilot_counts'][split]//3 for g in self.config['grid_catalog']})

    def test_full_plan_crosses_family_and_grid(self):
        full=build_plan(self.config,'full')
        self.assertEqual(len(full['entries']),3400)
        pilot_seeds={e['geology_seed'] for e in build_plan(self.config,'pilot')['entries']}
        self.assertFalse(pilot_seeds & {e['geology_seed'] for e in full['entries']})
        for split in ('train','validation_id','test_id'):
            pairs={(e['family'],e['grid_name']) for e in full['entries'] if e['split']==split}
            self.assertEqual(len(pairs),9)
        self.assertTrue(all(e['family']=='lens' for e in full['entries'] if e['split']=='test_family'))
        cov=config_checks(self.config)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0)

    def test_real_pilot_reproduces(self):
        result=validate_dataset(self.root,self.config,reproduce=True)
        self.assertEqual(result['volumes'],45)
        self.assertEqual(result['fingerprints_unique'],45)

    def test_corruption_detected_without_touching_real_pilot(self):
        first=self.manifest['entries'][0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shutil.copytree(self.root/first['realization_id'],root/first['realization_id'])
            copy_entry=copy.deepcopy(first)
            plan=build_plan(self.config,'pilot')['entries'][0]
            validate_entry(root,copy_entry,plan,self.config,reproduce=False)
            target=root/copy_entry['arrays']['Ic']
            with target.open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):
                validate_entry(root,copy_entry,plan,self.config,reproduce=False)

    def test_stored_vs_simulated_nested_observations(self):
        first=self.manifest['entries'][0]
        volume,_=_read_saved(self.root,first)
        observed=variants(volume['Ic'],volume['valid_mask'],grid_for(first),first['evaluation_site_seed'])
        self.assertEqual(tuple(observed),tuple(policy()['evaluation_profile_counts']))
        self.assertEqual(len(observed[20]['sites']),20)
        self.assertEqual(observed[1]['sites'],observed[20]['sites'][:1])
        self.assertTrue(np.array_equal(observed[1]['observed_ic'][~observed[1]['observed_mask']],
                                       np.zeros_like(observed[1]['observed_ic'][~observed[1]['observed_mask']])))
        self.assertFalse(np.array_equal(observed[20]['observed_mask'],observed[1]['observed_mask']))

    def test_orphan_recovery_and_idempotent_resume(self):
        first=self.manifest['entries'][0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'pilot'
            initialize(root,self.config)
            shutil.copytree(self.root/first['realization_id'],root/first['realization_id'])
            (root/first['realization_id']/'_entry.json').write_text(json.dumps(first))
            self.assertEqual(produce(root,self.config,limit=0)['volumes'],1)
            self.assertEqual(produce(root,self.config,limit=0)['volumes'],1)
            self.assertEqual(len(json.loads((root/'manifest.json').read_text())['entries']),1)

if __name__=='__main__':unittest.main()
