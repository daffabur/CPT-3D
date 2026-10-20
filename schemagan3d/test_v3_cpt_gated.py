import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from synthetic3d.v2.core import Grid
from .train_v3 import (_atomic_checkpoint, _checkpoint_payload, _dataset_loader,
                       _models, _train_epoch, evaluate_epoch, load_training_checkpoint,
                       NumericalBestEarlyStopping, validate_training_config)
from .v3_adapter import V3Normalization, compute_train_normalization, xyz_to_czyx
from .v3_cpt_gated import CPTGatedResidualGenerator3D, gate_features_from_mask
from .v3_idw_guided import idw_normalized_from_observed

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'data/synthetic3d_v2/prototype_geotechnical_v3'
torch.set_num_threads(1)


class CPTGatedTests(unittest.TestCase):
    def test_features_physical_distance_count_depth_and_supported_grids(self):
        for nx,ny,nz in ((12,12,12),(16,16,16),(24,24,12)):
            grid=Grid(nx,ny,nz,2.,3.,1.,origin_x_m=100.,origin_y_m=50.,azimuth_deg=19.)
            for count in (1,2,3,5,8,12,20):
                mask=torch.zeros(1,nz,ny,nx)
                for x,y in [(x,y) for x in range(nx) for y in range(ny)][:count]:mask[:, :, y, x]=1
                features=gate_features_from_mask(mask,grid)
                self.assertEqual(features.shape,(2,nz,ny,nx))
                self.assertTrue(torch.allclose(features[0],torch.full_like(features[0],count/20.)))
                self.assertTrue(torch.equal(features[:,0],features[:,-1]))
                self.assertEqual(float(features[1,0,0,0]),0.)
                self.assertTrue(torch.isfinite(features).all())
        grid=Grid(12,12,12,2.,3.,1.)
        mask=torch.zeros(1,12,12,12);mask[:,:,0,0]=1
        features=gate_features_from_mask(mask,grid)
        diagonal=np.hypot(11*2.,11*3.)
        self.assertAlmostEqual(float(features[1,0,0,1]),2./diagonal,places=6)
        self.assertAlmostEqual(float(features[1,0,1,0]),3./diagonal,places=6)

    def test_gate_range_initialization_and_no_hidden_target_leakage(self):
        torch.manual_seed(14)
        grid=Grid(12,12,12,2.,2.,1.)
        mask=torch.zeros(1,12,12,12);mask[:,:,2,2]=1
        observed=torch.zeros_like(mask);observed[mask.bool()]=2.5
        stats=V3Normalization(2.,.5)
        idw=idw_normalized_from_observed(observed,mask,grid,stats)
        features=gate_features_from_mask(mask,grid)
        sparse=torch.zeros_like(mask);sparse[mask.bool()]=1.
        model=CPTGatedResidualGenerator3D(4).eval()
        with torch.inference_mode():
            gated,prediction,gate,raw=model.forward_with_gate(sparse[None],mask[None],idw[None],features[None])
        self.assertTrue(torch.all((gate>=0)&(gate<=1)))
        self.assertLess(float(torch.max(torch.abs(gate-.25))),.01)
        self.assertTrue(torch.equal(prediction[mask[None].bool()],sparse[None][mask[None].bool()]))
        self.assertTrue(torch.allclose(gated,gate*raw))
        altered=observed.clone();altered[~mask.bool()]=100000.
        idw_altered=idw_normalized_from_observed(altered,mask,grid,stats)
        self.assertTrue(torch.equal(idw,idw_altered))
        with torch.inference_mode():
            _,prediction_altered,gate_altered,_=model.forward_with_gate(sparse[None],mask[None],idw_altered[None],features[None])
        self.assertTrue(torch.equal(prediction,prediction_altered))
        self.assertTrue(torch.equal(gate,gate_altered))

    def test_fixture_optimizer_validation_checkpoint_resume_and_legacy_variants(self):
        stats=compute_train_normalization(FIXTURE)
        loader=_dataset_loader(FIXTURE,'train',stats,[0],None,1,14,False,model_variant='idw_guided_gated_residual')
        batch=next(iter(loader))
        self.assertEqual(batch['gate_features'].shape[1],2)
        config={'model_variant':'idw_guided_gated_residual','base_channels':4,'train_cpt_counts':[1,2,3,5,8,12,20],
                'primary_evaluation_cpt_count':5,'training_cpt_policy':'mixed_per_epoch'}
        torch.manual_seed(14)
        generator,discriminator=_models(config)
        self.assertEqual(_models({'base_channels':4})[0].enc1[0].in_channels,2)
        self.assertEqual(_models({'model_variant':'idw_guided_residual','base_channels':4})[0].enc1[0].in_channels,3)
        opt_g=torch.optim.Adam(generator.parameters(),lr=.0002)
        opt_d=torch.optim.Adam(discriminator.parameters(),lr=.0002)
        gate_before=[p.detach().clone() for p in generator.gating_network.parameters()]
        losses=_train_epoch(generator,discriminator,loader,opt_g,opt_d,100.,'idw_guided_gated_residual')
        self.assertTrue(all(np.isfinite(list(losses.values()))))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(gate_before,generator.gating_network.parameters())))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                            for p in generator.gating_network.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in generator.residual_generator.parameters()))
        weights_after=[p.detach().clone() for p in generator.parameters()]
        metrics=evaluate_epoch(generator,discriminator,loader,stats,100.,'idw_guided_gated_residual')
        self.assertTrue(np.isfinite(metrics['hidden_rmse_physical_ic']))
        for key in ('gate_hidden_valid_mean','gate_hidden_valid_std','gate_hidden_valid_p10',
                    'gate_hidden_valid_p50','gate_hidden_valid_p90'):
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertGreaterEqual(metrics['gate_hidden_valid_p10'],0.)
        self.assertLessEqual(metrics['gate_hidden_valid_p90'],1.)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(weights_after,generator.parameters())))
        identity={'train_ids_sha256':'fixture'}
        stop=NumericalBestEarlyStopping(4,.005);stop.update(.5,1)
        payload=_checkpoint_payload(1,stop,config,identity,stats,generator,discriminator,opt_g,opt_d)
        self.assertTrue(any(k.startswith('gating_network.') for k in payload['generator_state']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.pt';_atomic_checkpoint(path,payload)
            restored_g,restored_d=_models(config)
            restored_og=torch.optim.Adam(restored_g.parameters(),lr=.0002)
            restored_od=torch.optim.Adam(restored_d.parameters(),lr=.0002)
            epoch,restored_stop=load_training_checkpoint(path,config,identity,stats,restored_g,restored_d,restored_og,restored_od)
            self.assertEqual(epoch,1);self.assertEqual(restored_stop.best_rmse,.5)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(generator.parameters(),restored_g.parameters())))
            self.assertEqual(opt_g.state_dict()['state'].keys(),restored_og.state_dict()['state'].keys())
            # The next fixture optimizer step after resume must match uninterrupted training.
            _train_epoch(generator,discriminator,loader,opt_g,opt_d,100.,'idw_guided_gated_residual')
            _train_epoch(restored_g,restored_d,loader,restored_og,restored_od,100.,'idw_guided_gated_residual')
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(generator.parameters(),restored_g.parameters())))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_training_checkpoint(path,{**config,'model_variant':'idw_guided_residual'},identity,stats,restored_g,restored_d,restored_og,restored_od)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_training_checkpoint(path,config,{'train_ids_sha256':'changed'},stats,restored_g,restored_d,restored_og,restored_od)

    def test_frozen_pilot_config(self):
        config=json.loads((ROOT/'schemagan3d/v3_cpt_gated_pilot_config.json').read_text())
        validate_training_config(config)
        self.assertEqual(config['subset_ids_path'],'results/schemagan3d_v3_pilot3_scaled_20e/subset_ids.json')
        self.assertEqual(config['output_dir'],'results/schemagan3d_v3_gated_pilot_20e')


if __name__=='__main__':unittest.main()
