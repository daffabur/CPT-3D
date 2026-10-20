"""Final 54-volume pilot acceptance and non-destructive production checks."""
import json
import copy
import shutil
import tempfile
from collections import Counter
from pathlib import Path
import unittest
import numpy as np
from .v3_dataset_production import CONFIG_PATH,build_plan,validate_dataset,produce,initialize,_read_saved,grid_for
from .v3_adapter import compute_train_normalization,V3VolumeDataset,HomogeneousShapeBatchSampler
from .v3_observation_policy import variants
from .train_v3 import dry_run

FINAL_CONFIG=Path(__file__).with_name('v3_final_pilot_config.json')

class FinalPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(FINAL_CONFIG.read_text())
        cls.old_config=json.loads(CONFIG_PATH.read_text())
        cls.root=Path(cls.config['output_root'])
        cls.entries=json.loads((cls.root/'manifest.json').read_text())['entries']

    def test_exact_family_grid_cells_and_holdouts(self):
        plan=build_plan(self.config,'pilot')['entries']
        self.assertEqual(len(plan),54)
        for split,want in (('train',2),('validation_id',1),('test_id',1)):
            counts=Counter((e['family'],e['grid_name']) for e in plan if e['split']==split)
            self.assertEqual(set(counts),{(f,g) for f in self.config['id_families'] for g in ('g12','g16','g24')})
            self.assertEqual(set(counts.values()),{want})
        for split,tag,family in (('test_family','base','lens'),('test_geometry_parameter','high_geometry','inclined'),('test_response_parameter','high_response','horizontal')):
            items=[e for e in plan if e['split']==split]
            self.assertEqual(Counter(e['grid_name'] for e in items),{'g12':2,'g16':2,'g24':2})
            self.assertEqual({e['family'] for e in items},{family})
            self.assertTrue(all(e['geometry_tag']==tag if split=='test_geometry_parameter' else e['response_tag']==tag for e in items))

    def test_seed_and_lineage_disjoint(self):
        final=build_plan(self.config,'pilot')['entries']
        other=build_plan(self.old_config,'pilot')['entries']+build_plan(self.old_config,'full')['entries']
        for key in ('geology_seed','material_seed','seed','site_seed','evaluation_site_seed','lineage_group'):
            self.assertFalse({e[key] for e in final}&{e[key] for e in other},key)
        self.assertEqual(len({e['lineage_group'] for e in final}),54)

    def test_reproduction_and_physical_consistency(self):
        result=validate_dataset(self.root,self.config,reproduce=True)
        self.assertEqual(result['volumes'],54)
        self.assertEqual(result['fingerprints_unique'],54)
        self.assertTrue(result['reproduced'])

    def test_nested_cpt_and_profile_counts(self):
        self.assertEqual({e['n_cpt'] for e in self.entries},{1,2,3,5,8,12,20})
        e=self.entries[0];v,o=_read_saved(self.root,e)
        patterns=variants(v['Ic'],v['valid_mask'],grid_for(e),e['evaluation_site_seed'])
        self.assertEqual(patterns[1]['sites'],patterns[20]['sites'][:1])
        self.assertEqual(len(patterns[20]['sites']),20)
        self.assertTrue(np.array_equal(patterns[1]['observed_ic'][~patterns[1]['observed_mask']],np.zeros_like(patterns[1]['observed_ic'][~patterns[1]['observed_mask']])))

    def test_adapter_normalization_and_batching(self):
        stats=compute_train_normalization(self.root)
        self.assertEqual(len(stats.source_train_ids),18)
        ds=V3VolumeDataset(self.root,split='train',stats=stats)
        sampler=HomogeneousShapeBatchSampler(ds,2,group_spacing=True)
        self.assertEqual(len(ds),18)
        for indices in sampler:
            self.assertEqual(len({ds.shapes[i] for i in indices}),1)
            self.assertEqual(len({ds.spacings[i] for i in indices}),1)

    def test_crash_recovery_corruption_and_config_freeze(self):
        e=self.entries[0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'final_pilot'
            initialize(root,self.config)
            shutil.copytree(self.root/e['realization_id'],root/e['realization_id'])
            recovered=produce(root,self.config,limit=0)
            self.assertEqual(recovered['volumes'],1)
            self.assertEqual(recovered['new_volumes'],0)
            changed=copy.deepcopy(self.config);changed['seed_namespace']+=1
            with self.assertRaisesRegex(ValueError,'altered config'):
                produce(root,changed,limit=0)
            with (root/e['arrays']['Ic']).open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):
                validate_dataset(root,self.config,reproduce=False)

    def test_dry_run_and_resume(self):
        cfg={'mode':'dry_run_only','dataset_root':str(self.root),'train_split':'train','validation_split':'validation_id',
             'base_channels':4,'batch_size':2,'reconstruction_weight':100.,'torch_seed':19,'batch_seed':20261019,
             'evaluation':{'profile_counts':self.config['evaluation_cpt_counts']}}
        result=dry_run(cfg,validator=lambda root:validate_dataset(root,self.config,reproduce=False))
        self.assertTrue(result['model_weights_unchanged'])
        self.assertTrue(result['generator_gradients_finite'] and result['discriminator_gradients_finite'])
        self.assertEqual(result['observed_max_error_physical'],0.)
        resumed=produce(self.root,self.config,limit=0)
        self.assertEqual(resumed['new_volumes'],0)
        self.assertEqual(resumed['volumes'],54)

if __name__=='__main__':unittest.main()
