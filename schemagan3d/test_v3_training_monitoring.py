"""Monitoring tests never take an optimizer step on the 24K dataset."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .models import Generator3D, Discriminator3D
from .v3_adapter import V3Normalization,V3VolumeDataset,compute_train_normalization,load_manifest
from .train_v3 import (EarlyStopping,HISTORY_FIELDS,_atomic_checkpoint,_batch_metrics_physical,
                       _checkpoint_payload,_split_metadata,balanced_indices,load_training_checkpoint,
                       monitoring_dry_run,prepare_monitoring,render_learning_curves,evaluate_cpt_sweep,
                       train_pilot,validate_resume_history,write_history)

ROOT=Path(__file__).resolve().parents[1]/'data/synthetic3d_v2/schemagan_v3_final_pilot'
PILOT_CONFIG=Path(__file__).with_name('v3_pilot_training_config.json')


class MonitoringTests(unittest.TestCase):
    def test_physical_metrics_and_gap(self):
        stats=V3Normalization(2.,2.)
        prediction=torch.zeros((2,1,1,1,2))
        target=torch.tensor([[[[[0.,.5]]]],[[[[0.,1.]]]]])
        mask=torch.tensor([[[[[1.,0.]]]],[[[[1.,0.]]]]])
        batch={'target':target,'mask':mask,'valid_mask':torch.ones_like(mask),
               'observed_ic_physical':torch.zeros_like(mask)}
        metrics=_batch_metrics_physical(prediction,batch,stats)
        self.assertEqual([m['rmse'] for m in metrics],[1.,2.])
        self.assertEqual([m['mae'] for m in metrics],[1.,2.])
        self.assertEqual(sum(m['rmse'] for m in metrics)/2,1.5)
        self.assertEqual(2.-1.5,.5)  # validation − train gap in physical Ic.

    def test_early_stopping_best_epoch(self):
        stop=EarlyStopping(patience=2,min_delta=.01)
        self.assertEqual(stop.update(1.,1),(True,False))
        self.assertEqual(stop.update(.995,2),(False,False))
        self.assertEqual(stop.update(.98,3),(True,False))
        self.assertEqual(stop.update(.979,4),(False,False))
        self.assertEqual(stop.update(.978,5),(False,True))
        self.assertEqual((stop.best_epoch,stop.best_rmse),(3,.98))
        self.assertEqual(EarlyStopping.from_state(stop.state()).state(),stop.state())

    def test_deterministic_nested_evaluation_counts(self):
        stats=compute_train_normalization(ROOT)
        for count in (1,2,3,5,8,12,20):
            ds=V3VolumeDataset(ROOT,split='train',stats=stats,evaluation_cpt_count=count,indices=[0])
            a,b=ds[0],ds[0]
            self.assertTrue(torch.equal(a['mask'],b['mask']))
            self.assertEqual(int(a['mask'].sum()/a['target'].shape[1]),count)
            self.assertTrue(torch.all(a['sparse'][a['mask']==0]==0))
        one=V3VolumeDataset(ROOT,split='train',stats=stats,evaluation_cpt_count=1,indices=[0])[0]
        twenty=V3VolumeDataset(ROOT,split='train',stats=stats,evaluation_cpt_count=20,indices=[0])[0]
        self.assertTrue(torch.all(twenty['mask'][one['mask'].bool()]==1))

    def _fixture_config(self,tmp):
        cfg=json.loads(PILOT_CONFIG.read_text());cfg['dataset_root']=str(ROOT)
        cfg['output_dir']=str(Path(tmp)/'training')
        manifest=json.loads((ROOT/'manifest.json').read_text())
        cfg['expected_plan_sha256']=manifest['plan_sha256']
        cfg['expected_split_counts']={s:sum(e['split']==s for e in manifest['entries'])
                                      for s in {e['split'] for e in manifest['entries']}}
        cfg['training_per_cell']=2;cfg['evaluation_per_cell']=1;cfg['base_channels']=2
        stats=compute_train_normalization(ROOT)
        ids=stats.source_train_ids
        meta={'plan_sha256':manifest['plan_sha256'],'source_split':'train','source_volume_count':len(ids),
              'source_train_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
              'mean':stats.mean,'std':stats.std}
        path=Path(tmp)/'normalization.json';path.write_text(json.dumps(meta));cfg['normalization_metadata']=str(path)
        return cfg

    def test_read_only_monitoring_dry_run_and_leakage_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self._fixture_config(tmp)
            root,stats,identity,train,train_eval,val_eval=prepare_monitoring(cfg)
            self.assertEqual(len(train),18);self.assertEqual(len(train_eval),9);self.assertEqual(len(val_eval),9)
            self.assertTrue(set(train_eval).issubset(train))
            self.assertEqual(identity['split_counts'],cfg['expected_split_counts'])
            result=monitoring_dry_run(cfg)
            self.assertFalse(result['optimizer_created']);self.assertFalse(result['optimizer_step'])
            self.assertFalse(result['checkpoint_written'])
            self.assertEqual(result['normalization']['source_split'],'train')
            self.assertAlmostEqual(result['generalization_gap_rmse_physical_ic'],
                                   result['validation_id']['hidden_rmse_physical_ic']-result['train']['hidden_rmse_physical_ic'])
            bad=copy.deepcopy(cfg);bad['expected_split_counts']['test_id']+=1
            with self.assertRaisesRegex(ValueError,'split counts'):prepare_monitoring(bad)
            bad=copy.deepcopy(cfg);meta=json.loads(Path(cfg['normalization_metadata']).read_text())
            meta['mean']+=.1;Path(cfg['normalization_metadata']).write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'normalization metadata'):prepare_monitoring(bad)

    def test_lineage_cross_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[{'split':s,'lineage_group':'same' if s in ('train','validation_id') else s,
                                      'realization_id':s} for s in ('train','validation_id','test_id','test_family',
                                                                    'test_geometry_parameter','test_response_parameter')]
            (root/'manifest.json').write_text(json.dumps({'entries':entries,'plan_sha256':'x'}))
            stats=V3Normalization(2.,1.,source_train_ids=('train',))
            with self.assertRaisesRegex(ValueError,'lineage crosses'): _split_metadata(root,{'train_split':'train','validation_split':'validation_id'},stats)

    def test_checkpoint_history_curves_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config={'mode':'fixture'};identity={'plan_sha256':'fixture','train_ids_sha256':'fixture-ids'}
            stats=V3Normalization(2.,.5)
            g=Generator3D(2);d=Discriminator3D(2)
            opt_g=torch.optim.Adam(g.parameters(),lr=.001);opt_d=torch.optim.Adam(d.parameters(),lr=.001)
            stop=EarlyStopping(2,.01);stop.update(.9,1)
            torch.manual_seed(123)
            payload=_checkpoint_payload(1,stop,config,identity,stats,g,d,opt_g,opt_d)
            _atomic_checkpoint(root/'best.pt',payload);_atomic_checkpoint(root/'last.pt',payload)
            before=[p.detach().clone() for p in g.parameters()]
            expected=torch.rand(1)
            with torch.no_grad():next(g.parameters()).add_(1)
            completed,restored=load_training_checkpoint(root/'last.pt',config,identity,stats,g,d,opt_g,opt_d)
            self.assertEqual(completed,1);self.assertEqual(restored.best_epoch,1)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(before,g.parameters())))
            self.assertTrue(torch.equal(expected,torch.rand(1)))
            row=dict.fromkeys(HISTORY_FIELDS,0)
            row.update(epoch=1,train_hidden_rmse_physical_ic=.7,validation_hidden_rmse_physical_ic=.9,
                       generalization_gap_rmse_physical_ic=.2,generator_loss_total=2.,
                       discriminator_loss=1.,stop_reason='')
            write_history(root,[row],config,identity)
            rows,next_epoch=validate_resume_history(root,config,identity,completed,restored)
            self.assertEqual(next_epoch,2);self.assertEqual(rows,[row])
            render_learning_curves(rows,root/'learning_curves.png')
            self.assertGreater((root/'learning_curves.png').stat().st_size,1000)
            with (root/'training_history.csv').open() as stream:
                self.assertEqual(list(__import__('csv').DictReader(stream))[0]['epoch'],'1')
            history=json.loads((root/'training_history.json').read_text());history['rows'][-1]['epoch']=2
            (root/'training_history.json').write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError,'resume history'):validate_resume_history(root,config,identity,1,restored)
            history['rows'][-1]['epoch']=1
            (root/'training_history.json').write_text(json.dumps(history))
            (root/'training_history.csv').write_text('epoch\n2\n')
            with self.assertRaisesRegex(ValueError,'resume history'):validate_resume_history(root,config,identity,1,restored)
            with self.assertRaisesRegex(ValueError,'checkpoint config'):
                load_training_checkpoint(root/'last.pt',{'mode':'different'},identity,stats,g,d,opt_g,opt_d)

    def test_best_checkpoint_and_patience_without_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self._fixture_config(tmp);cfg['epochs']=5;cfg['patience']=2;cfg['min_delta']=.01
            values=iter((1.,.8,.805,.806))
            def fake_eval(g,d,loader,stats,weight):
                split=loader.dataset.entries[0]['split']
                rmse=.5 if split=='train' else next(values)
                return {'hidden_rmse_physical_ic':rmse,'hidden_mae_physical_ic':rmse/2,
                        'generator_loss_total':2.,'discriminator_loss':1.,
                        'reconstruction_loss_normalized_ic':.1}
            loss={'generator_loss_total':2.,'discriminator_loss':1.,'reconstruction_loss_normalized_ic':.1}
            with patch('schemagan3d.train_v3._train_epoch',return_value=loss) as train_mock:
                with patch('schemagan3d.train_v3.evaluate_epoch',side_effect=fake_eval):
                    result=train_pilot(cfg)
            self.assertEqual(train_mock.call_count,4)
            self.assertEqual((result['best_epoch'],result['epochs_completed'],result['stop_reason']),
                             (2,4,'patience_exhausted'))
            root=Path(cfg['output_dir'])
            best=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(best['epoch'],2);self.assertEqual(last['epoch'],4)
            self.assertEqual(best['best_validation_rmse_physical_ic'],.8)
            rows=json.loads((root/'training_history.json').read_text())['rows']
            self.assertEqual([r['generalization_gap_rmse_physical_ic'] for r in rows],
                             [1.-.5,.8-.5,.805-.5,.806-.5])
            self.assertTrue((root/'learning_curves.png').is_file())
            sweep=evaluate_cpt_sweep(cfg,'validation_id',counts=(1,5))
            self.assertEqual(set(sweep['metrics_by_cpt_count']),{'1','5'})
            self.assertEqual(sweep['metrics_by_cpt_count']['1']['volumes'],9)
            self.assertEqual(sweep['metrics_by_cpt_count'],
                             evaluate_cpt_sweep(cfg,'validation_id',counts=(1,5))['metrics_by_cpt_count'])
            with self.assertRaisesRegex(ValueError,'test/holdout'):evaluate_cpt_sweep(cfg,'test_id',counts=(1,))

    def test_resume_starts_at_next_epoch_without_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self._fixture_config(tmp);cfg['epochs']=3;cfg['patience']=5;cfg['min_delta']=0.
            values=iter((1.,.9,.8))
            def fake_eval(g,d,loader,stats,weight):
                rmse=.5 if loader.dataset.entries[0]['split']=='train' else next(values)
                return {'hidden_rmse_physical_ic':rmse,'hidden_mae_physical_ic':rmse,
                        'generator_loss_total':2.,'discriminator_loss':1.,
                        'reconstruction_loss_normalized_ic':.1}
            loss={'generator_loss_total':2.,'discriminator_loss':1.,'reconstruction_loss_normalized_ic':.1}
            with patch('schemagan3d.train_v3._train_epoch',side_effect=[loss,loss,RuntimeError('fixture interrupt')]):
                with patch('schemagan3d.train_v3.evaluate_epoch',side_effect=fake_eval):
                    with self.assertRaisesRegex(RuntimeError,'fixture interrupt'):train_pilot(cfg)
            root=Path(cfg['output_dir'])
            self.assertEqual(torch.load(root/'last.pt',map_location='cpu',weights_only=False)['epoch'],2)
            with patch('schemagan3d.train_v3._train_epoch',return_value=loss) as train_mock:
                with patch('schemagan3d.train_v3.evaluate_epoch',side_effect=fake_eval):
                    result=train_pilot(cfg,resume=True)
            self.assertEqual(train_mock.call_count,1)
            self.assertEqual(result['epochs_completed'],3)
            rows=json.loads((root/'training_history.json').read_text())['rows']
            self.assertEqual([r['epoch'] for r in rows],[1,2,3])


if __name__=='__main__':unittest.main()
