"""Plan-only 24K tests, including deliberate split, seed, lineage and grid damage."""
import copy
import json
from pathlib import Path
import unittest
from collections import Counter
from .v3_24k_plan import CONFIG_PATH,build_plan,validate_config,validate_plan,estimate_resources
from .v3_dataset_production import build_plan as old_build_plan,digest,grid_for,settings

class Plan24KTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(CONFIG_PATH.read_text())
        cls.plan=build_plan(cls.config)

    def changed(self,index=0,**fields):
        p=copy.deepcopy(self.plan)
        p['entries'][index].update(fields)
        return p

    def test_counts_balance_determinism_and_prior_compatibility(self):
        result=validate_plan(self.plan,self.config,expected_hash=digest(self.plan))
        self.assertEqual(result['total'],24000)
        self.assertEqual(result['split_counts']['train'],16000)
        self.assertEqual(result['split_counts']['validation_id'],4000)
        self.assertEqual(sum(result['split_counts'][s] for s in ('test_id','test_family','test_geometry_parameter','test_response_parameter')),4000)
        self.assertTrue(result['deterministic_rebuild_equal'])
        self.assertEqual(digest(build_plan(self.config)),digest(self.plan))
        for e in self.plan['entries']:
            self.assertEqual(grid_for(e).nx,e['shape'][0])
        geo=next(e for e in self.plan['entries'] if e['split']=='test_geometry_parameter')
        response=next(e for e in self.plan['entries'] if e['split']=='test_response_parameter')
        self.assertAlmostEqual(settings(geo,self.config)[0]['common_slope_fraction'],.16)
        self.assertEqual(settings(response,self.config)[1]['log_covariance'],[[.32**2,-.012],[-.012,.30**2]])

    def test_old_3400_and_final_pilot_plan_unchanged(self):
        root=Path(__file__).parent
        old=json.loads((root/'v3_dataset_config.json').read_text())
        final=json.loads((root/'v3_final_pilot_config.json').read_text())
        self.assertEqual(digest(old_build_plan(old,'full')),'fcd34f4380d6270c6013b63270a4c915ea7c4c5cf86b8de8e1cdae2c74e9c74b')
        self.assertEqual(len(old_build_plan(final,'pilot')['entries']),54)

    def test_reject_duplicate_seed_and_lineage(self):
        first=self.plan['entries'][0]
        with self.assertRaisesRegex(ValueError,'seed collision'):
            validate_plan(self.changed(1,geology_seed=first['geology_seed']),self.config,check_determinism=False)
        with self.assertRaisesRegex(ValueError,'lineage'):
            validate_plan(self.changed(1,lineage_group=first['lineage_group']),self.config,check_determinism=False)

    def test_reject_holdout_leakage_and_grid_damage(self):
        with self.assertRaisesRegex(ValueError,'ID family/tag leakage'):
            validate_plan(self.changed(family='lens'),self.config,check_determinism=False)
        with self.assertRaisesRegex(ValueError,'grid metadata'):
            validate_plan(self.changed(grid_id='g12' if self.plan['entries'][0]['grid_id']!='g12' else 'g24'),self.config,check_determinism=False)

    def test_reject_family_grid_imbalance(self):
        first=self.plan['entries'][0]
        replacement='g12' if first['grid_id']!='g12' else 'g24'
        altered=self.changed(grid_id=replacement,grid_name=replacement,
                             shape=next(g['shape'] for g in self.config['grid_catalog'] if g['name']==replacement),
                             spacing_m=next(g['spacing_m'] for g in self.config['grid_catalog'] if g['name']==replacement))
        with self.assertRaisesRegex(ValueError,'family-grid imbalance'):
            validate_plan(altered,self.config,check_determinism=False)

    def test_reject_wrong_split_and_hash(self):
        with self.assertRaisesRegex(ValueError,'entry count'):
            p=copy.deepcopy(self.plan);p['entries'].pop()
            validate_plan(p,self.config,check_determinism=False)
        with self.assertRaisesRegex(ValueError,'canonical SHA-256'):
            validate_plan(self.plan,self.config,expected_hash='0'*64,check_determinism=False)

    def test_reject_config_change_and_output_collision(self):
        c=copy.deepcopy(self.config);c['response_holdout']['log_qc_sd']=.33
        with self.assertRaisesRegex(ValueError,'response holdout'):
            validate_config(c)
        c=copy.deepcopy(self.config);c['output_root']='data/synthetic3d_v2/schemagan_v3_final_pilot'
        with self.assertRaisesRegex(ValueError,'unsafe output'):
            validate_config(c)

    def test_resource_projection_grid_weighted(self):
        b={'per_grid':{g:{'stored_bytes_mean':size,'generation_seconds_mean':.01} for g,size in
                       (('g12',100.),('g16',200.),('g24',300.))},
           'actual_allocated_kib_du':100,'payload_bytes_54':100000,'actual_file_count':1029,
           'production_function_end_to_end_seconds_including_manifest_and_integrity_check':1.,
           'process_wall_seconds_from_time_utility':2.,'manifest_write_seconds_for_54':.1,
           'full_validation_seconds_separate_process':.4,'temporary_g24_writer_peak_rss_bytes':200000000,
           'peak_rss_bytes_during_full_regeneration_and_validation':210000000,
           'metadata_only_manifest_simulation':[{'entries':800,'seconds':14.,'final_bytes':4000000}]}
        estimate=estimate_resources(self.plan,self.config,b)
        counts=estimate['grid_counts']
        self.assertEqual(sum(counts.values()),24000)
        self.assertEqual(estimate['projection_24k']['payload_bytes'],sum(counts[g]*b['per_grid'][g]['stored_bytes_mean'] for g in counts))
        self.assertEqual(estimate['projection_24k']['total_volume_files_including_sidecars'],456000)
        self.assertGreater(estimate['projection_24k']['manifest_rewrite_seconds_quadratic_metadata_simulation'],10000)

if __name__=='__main__':unittest.main()
