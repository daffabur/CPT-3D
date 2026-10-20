import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from synthetic3d.v2.core import Grid
from .prepare_v3_small_data import DEFAULT_ROOT,validate_small_manifest
from .v3_adapter import V3VolumeDataset,compute_train_normalization,HomogeneousShapeBatchSampler
from .v3_eval_protocol import PROFILE_COUNTS,nested_sites,idw_physical,sparse_from_sites,valid_metrics
from .v3_losses import hidden_valid_l1,generator_objective
from .train_v3 import dry_run
from .models import Discriminator3D

CONFIG=Path(__file__).with_name('v3_experiment_config.json')

class TrainingPrepTests(unittest.TestCase):
    def test_actual_six_splits_and_reproduction(self):
        summary=validate_small_manifest(DEFAULT_ROOT)
        self.assertEqual(summary['volumes'],16)
        self.assertEqual(set(summary['splits']),{'train','validation_id','test_id','test_family','test_geometry_parameter','test_response_parameter'})
        self.assertTrue(all(v>0 for v in summary['splits'].values()))
    def test_train_only_stats_and_loader(self):
        stats=compute_train_normalization(DEFAULT_ROOT)
        self.assertEqual(len(stats.source_train_ids),3)
        train=V3VolumeDataset(DEFAULT_ROOT,split='train',stats=stats)
        for i in range(len(train)):
            x=train[i]
            self.assertEqual(x['sparse'].shape,x['target'].shape)
            self.assertTrue(torch.all(x['sparse'][x['mask']==0]==0))
            self.assertTrue(torch.all(x['valid_mask']==1))
            self.assertGreaterEqual(int(x['mask'].sum()/x['target'].shape[1]),1)
        response=V3VolumeDataset(DEFAULT_ROOT,split='test_response_parameter',stats=stats)
        self.assertEqual(len(response),2)
        with self.assertRaisesRegex(ValueError,'provenance'):
            V3VolumeDataset(DEFAULT_ROOT,split='test_id',stats=type(stats)(0.,1.,source_train_ids=('test-id',)))
    def test_sampler_shape_and_spacing(self):
        ds=SimpleNamespace(shapes=[(12,12,12),(12,12,12),(12,12,12),(16,16,16)],
                           spacings=[(2.,2.,1.),(2.,2.,1.),(1.5,1.5,1.),(2.,2.,1.)])
        sampler=HomogeneousShapeBatchSampler(ds,2,group_spacing=True)
        self.assertEqual(list(sampler),[[0,1],[2],[3]])
        self.assertEqual(list(HomogeneousShapeBatchSampler(ds,2,group_spacing=False)),[[0,1],[2],[3]])
        stats=compute_train_normalization(DEFAULT_ROOT)
        train=V3VolumeDataset(DEFAULT_ROOT,split='train',stats=stats)
        for batch in DataLoader(train,batch_sampler=HomogeneousShapeBatchSampler(train,2,group_spacing=True)):
            self.assertTrue(torch.all(batch['spacing_m']==batch['spacing_m'][0]))
    def test_manual_per_volume_masked_l1(self):
        pred=torch.tensor([[[[[100.,1.,3.,float('nan')]]]],[[[[100.,2.,4.,6.]]]]],requires_grad=True)
        target=torch.zeros_like(pred);mask=torch.tensor([[[[[1,0,0,0]]]],[[[[1,0,0,0]]]]],dtype=torch.float32)
        valid=torch.tensor([[[[[1,1,1,0]]]],[[[[1,1,1,1]]]]],dtype=torch.float32)
        loss=hidden_valid_l1(pred,target,mask,valid)
        self.assertEqual(float(loss),3.0) # means 2 and 4, then average
        loss.backward();self.assertTrue(torch.isfinite(pred.grad).all());self.assertEqual(float(pred.grad[0,0,0,0,0]),0.)
        self.assertEqual(float(pred.grad[0,0,0,0,3]),0.)
    def test_zero_hidden_and_shape_error(self):
        p=torch.zeros((1,1,8,8,8));all_observed=torch.ones_like(p)
        with self.assertRaisesRegex(ValueError,'hidden valid'):hidden_valid_l1(p,p,all_observed,all_observed)
        with self.assertRaisesRegex(ValueError,'shape'):hidden_valid_l1(p,p[:,:,:4],p,p)
    def test_adversarial_objective_numerical(self):
        logits=torch.zeros((1,1,2,2,2));p=torch.ones((1,1,8,8,8));t=torch.zeros_like(p);mask=torch.zeros_like(p);valid=torch.ones_like(p)
        total,adv,rec=generator_objective(logits,p,t,mask,valid,reconstruction_weight=100.)
        self.assertAlmostEqual(float(adv),float(np.log(2)),places=6)
        self.assertEqual(float(rec),1.)
        self.assertAlmostEqual(float(total),100+float(np.log(2)),places=5)
    def test_discriminator_patch_shapes(self):
        model=Discriminator3D(base_channels=4).eval()
        with torch.inference_mode():
            for shape,expected in [((12,12,12),(1,1,1,1,1)),((16,16,16),(1,1,2,2,2)),((12,24,24),(1,1,1,3,3))]:
                a=torch.zeros((1,1,*shape))
                self.assertEqual(tuple(model(a,a,a).shape),expected)
    def test_cpt_counts_nested_and_capacity(self):
        grid=Grid(8,8,8,2,2,1);valid=np.ones((8,8,8),bool)
        sites=nested_sites(grid,42,valid,20);self.assertEqual(len(sites),20)
        for n in PROFILE_COUNTS:
            self.assertEqual(sites[:n],nested_sites(grid,42,valid,n))
        small=Grid(4,4,8,2,2,1);self.assertEqual(len(nested_sites(small,42,np.ones((4,4,8),bool),20)),16)
    def test_physical_idw_uses_observations_only(self):
        a=Grid(4,4,8,1,1,1);b=Grid(4,4,8,2,1,1)
        observed=np.zeros((4,4,8),bool);observed[0,0,:]=True;observed[3,0,:]=True
        values=np.full((4,4,8),np.nan,dtype=np.float32);values[0,0,:]=1.;values[3,0,:]=3.
        x=idw_physical(values,observed,a);y=idw_physical(values,observed,b)
        self.assertTrue(np.isfinite(x).all());self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.array_equal(x[observed],values[observed]))
        self.assertNotAlmostEqual(float(x[1,1,0]),float(y[1,1,0]),places=4)
        hidden_values=values.copy();hidden_values[~observed]=9999.
        self.assertTrue(np.array_equal(x,idw_physical(hidden_values,observed,a)))
        sparse=sparse_from_sites(values,observed);self.assertTrue(np.all(sparse[~observed]==0))
        metrics=valid_metrics(x,np.ones_like(x),observed,np.ones_like(observed))
        self.assertEqual(metrics['hidden_count'],(4*4-2)*8)
    def test_dry_run_gradients_no_optimizer_step(self):
        config=json.loads(CONFIG.read_text());out=dry_run(config)
        self.assertTrue(out['model_weights_unchanged']);self.assertFalse(out['optimizer_created']);self.assertFalse(out['optimizer_step']);self.assertFalse(out['checkpoint_written'])
        self.assertTrue(out['generator_gradients_finite']);self.assertTrue(out['discriminator_gradients_finite'])
        self.assertEqual(out['observed_max_error_physical'],0.)
        self.assertEqual(out['generator_output_shape'],out['train_batch_shape'])
        with self.assertRaisesRegex(ValueError,'dry_run_only'):dry_run({**config,'mode':'train'})

if __name__=='__main__':unittest.main()
