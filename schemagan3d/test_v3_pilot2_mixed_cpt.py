"""Pilot 2 sampling/selection tests use fixture data and mocked training epochs."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from . import test_v3_training_monitoring as monitoring_tests
from .v3_adapter import V3VolumeDataset,compute_train_normalization,HomogeneousShapeBatchSampler
from .train_v3 import (NumericalBestEarlyStopping,EarlyStopping,mixed_cpt_counts,
                       train_pilot,prepare_monitoring)

class Pilot2Tests(unittest.TestCase):
    def test_mixed_counts_deterministic_balanced_and_resume_order(self):
        a=mixed_cpt_counts(2000,[1,2,3,5,8,12,20],14,3)
        self.assertEqual(a,mixed_cpt_counts(2000,[1,2,3,5,8,12,20],14,3))
        self.assertNotEqual(a,mixed_cpt_counts(2000,[1,2,3,5,8,12,20],14,4))
        self.assertEqual(set(a),{1,2,3,5,8,12,20})
        self.assertLessEqual(max(a.count(c) for c in set(a))-min(a.count(c) for c in set(a)),1)

    def test_per_sample_masks_and_homogeneous_batches(self):
        stats=compute_train_normalization(monitoring_tests.ROOT)
        counts=[1,2,3,5,8,12,20]
        ds=V3VolumeDataset(monitoring_tests.ROOT,split='train',stats=stats,indices=list(range(7)),cpt_counts_by_index=counts)
        for i,count in enumerate(counts):
            item=ds[i]
            self.assertEqual(int(item['mask'].sum()/item['mask'].shape[1]),count)
            self.assertTrue(torch.all(item['sparse'][item['mask']==0]==0))
        sampler=HomogeneousShapeBatchSampler(ds,2,shuffle=True,seed=14,group_spacing=True)
        self.assertEqual(list(sampler),list(HomogeneousShapeBatchSampler(ds,2,shuffle=True,seed=14,group_spacing=True)))
        for batch in sampler:
            self.assertEqual(len({(ds.shapes[i],ds.spacings[i]) for i in batch}),1)

    def test_numeric_best_separate_from_patience_and_old_checkpoint(self):
        stop=NumericalBestEarlyStopping(2,.01)
        self.assertEqual(stop.update(1.,1),(True,False))
        self.assertEqual(stop.update(.8,2),(True,False))
        self.assertEqual(stop.update(.795,3),(True,False))
        self.assertEqual(stop.update(.794,4),(True,True))
        self.assertEqual((stop.best_epoch,stop.best_rmse,stop.early_stop_reference_rmse,stop.bad_epochs),(4,.794,.8,2))
        self.assertEqual(EarlyStopping.from_state(stop.state()).state(),stop.state())
        old=EarlyStopping(2,.01);old.update(.9,1)
        self.assertEqual(type(EarlyStopping.from_state(old.state())),EarlyStopping)

    def test_saved_subset_identity_and_no_test_loader(self):
        config=json.loads(Path('schemagan3d/v3_pilot2_mixed_cpt_config.json').read_text())
        _,_,_,train,train_eval,val=prepare_monitoring(config)
        self.assertEqual((len(train),len(train_eval),len(val)),(2000,400,400))
        saved=json.loads(Path(config['subset_ids_path']).read_text())
        self.assertEqual(len(saved['ids']['train']),len(train))
        bad=config.copy()
        with tempfile.TemporaryDirectory() as tmp:
            copy=json.loads(Path(config['subset_ids_path']).read_text())
            copy['ids']['train'][0]='tampered'
            p=Path(tmp)/'subset.json';p.write_text(json.dumps(copy));bad['subset_ids_path']=str(p)
            with self.assertRaisesRegex(ValueError,'subset IDs'):prepare_monitoring(bad)

    def test_scaled_subset_contains_parent_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=monitoring_tests.MonitoringTests()._fixture_config(tmp)
            root,_,identity,train,train_eval,val=prepare_monitoring(cfg)
            entries=json.loads((root/'manifest.json').read_text())['entries']
            split_entries={split:[e for e in entries if e['split']==split] for split in ('train','validation_id')}
            ids={name:[split_entries[split][i]['realization_id'] for i in selected[:3]]
                 for name,split,selected in [('train','train',train),('train_evaluation','train',train_eval),
                                              ('validation_id','validation_id',val)]}
            parent={'dataset_identity':identity,'ids':ids,
                    'sha256':{k:hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                              for k,v in ids.items()}}
            path=Path(tmp)/'parent.json';path.write_text(json.dumps(parent))
            cfg['parent_subset_ids_path']=str(path)
            prepare_monitoring(cfg)
            parent['ids']['train'][0]='tampered';path.write_text(json.dumps(parent))
            with self.assertRaisesRegex(ValueError,'parent subset ID hash'):prepare_monitoring(cfg)

    def test_numeric_checkpoint_during_mock_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=monitoring_tests.MonitoringTests()._fixture_config(tmp)
            cfg.update(training_cpt_policy='mixed_per_epoch',epochs=5,patience=2,min_delta=.01)
            values=iter((1.,.8,.795,.794))
            def fake_eval(g,d,loader,stats,weight):
                rmse=.5 if loader.dataset.entries[0]['split']=='train' else next(values)
                return {'hidden_rmse_physical_ic':rmse,'hidden_mae_physical_ic':rmse,
                        'generator_loss_total':2.,'discriminator_loss':1.,
                        'reconstruction_loss_normalized_ic':.1}
            loss={'generator_loss_total':2.,'discriminator_loss':1.,'reconstruction_loss_normalized_ic':.1}
            with patch('schemagan3d.train_v3._train_epoch',return_value=loss):
                with patch('schemagan3d.train_v3.evaluate_epoch',side_effect=fake_eval):
                    result=train_pilot(cfg)
            self.assertEqual((result['epochs_completed'],result['best_epoch'],result['best_validation_rmse_physical_ic']),(4,4,.794))
            output=Path(cfg['output_dir'])
            best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
            self.assertEqual(best['epoch'],4)
            self.assertEqual(best['early_stopping']['early_stop_reference_rmse'],.8)
            history=json.loads((output/'training_history.json').read_text())['rows']
            histograms=[json.loads(row['training_cpt_histogram_json']) for row in history]
            self.assertEqual({sum(h.values()) for h in histograms},{18})
            self.assertTrue(all(set(h)=={'1','2','3','5','8','12','20'} and max(h.values())-min(h.values())<=1
                                for h in histograms))
            self.assertEqual([row['early_stopping_bad_epochs'] for row in history],[0,0,1,2])

    def test_mixed_resume_uses_next_epoch_sampling_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=monitoring_tests.MonitoringTests()._fixture_config(tmp)
            cfg.update(training_cpt_policy='mixed_per_epoch',epochs=3,patience=5,min_delta=0.)
            seen=[]
            def fake_train(g,d,loader,opt_g,opt_d,weight):
                seen.append(loader.dataset.cpt_counts_by_index)
                if len(seen)==3:raise RuntimeError('fixture interruption')
                return {'generator_loss_total':2.,'discriminator_loss':1.,
                        'reconstruction_loss_normalized_ic':.1}
            values=iter((1.,.9,.8))
            def fake_eval(g,d,loader,stats,weight):
                rmse=.5 if loader.dataset.entries[0]['split']=='train' else next(values)
                return {'hidden_rmse_physical_ic':rmse,'hidden_mae_physical_ic':rmse,
                        'generator_loss_total':2.,'discriminator_loss':1.,
                        'reconstruction_loss_normalized_ic':.1}
            with patch('schemagan3d.train_v3._train_epoch',side_effect=fake_train):
                with patch('schemagan3d.train_v3.evaluate_epoch',side_effect=fake_eval):
                    with self.assertRaisesRegex(RuntimeError,'fixture interruption'):train_pilot(cfg)
            expected=tuple(mixed_cpt_counts(18,cfg['train_cpt_counts'],cfg['random_seed'],3))
            self.assertEqual(seen[-1],expected)
            seen.clear()
            with patch('schemagan3d.train_v3._train_epoch',side_effect=fake_train):
                with patch('schemagan3d.train_v3.evaluate_epoch',side_effect=fake_eval):
                    result=train_pilot(cfg,resume=True)
            self.assertEqual(seen,[expected])
            self.assertEqual(result['epochs_completed'],3)

if __name__=='__main__':unittest.main()
