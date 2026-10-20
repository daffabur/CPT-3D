"""Fixture-only checks for reconstruction-only IDW residual objectives."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from .train_v3 import (_atomic_checkpoint, _checkpoint_payload, _dataset_loader,
                       _models, _train_epoch, evaluate_epoch, load_training_checkpoint,
                       NumericalBestEarlyStopping, validate_training_config)
from .v3_adapter import compute_train_normalization, restore_observed
from .v3_losses import hidden_valid_l1, hidden_valid_mse

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'data/synthetic3d_v2/prototype_geotechnical_v3'
torch.set_num_threads(1)


class ObjectiveAblationTests(unittest.TestCase):
    def test_mask_reduction_and_gradient(self):
        pred=torch.tensor([[[[[2.,4.,100.]]]],[[[[3.,10.,100.]]]]],requires_grad=True)
        target=torch.zeros_like(pred)
        observed=torch.tensor([[[[[1.,0.,0.]]]],[[[[1.,0.,0.]]]]])
        valid=torch.tensor([[[[[1.,1.,0.]]]],[[[[1.,1.,0.]]]]])
        self.assertAlmostEqual(hidden_valid_l1(pred,target,observed,valid).item(),7.)
        self.assertAlmostEqual(hidden_valid_mse(pred,target,observed,valid).item(),58.)
        hidden_valid_mse(pred,target,observed,valid).backward()
        self.assertEqual(pred.grad[0,0,0,0,0].item(),0.)
        self.assertEqual(pred.grad[0,0,0,0,2].item(),0.)
        self.assertGreater(pred.grad[0,0,0,0,1].item(),0.)

    def test_objectives_one_step_no_discriminator_update_resume_identity(self):
        config=json.loads((ROOT/'schemagan3d/v3_idw_guided_pilot_config.json').read_text())
        stats=compute_train_normalization(FIXTURE)
        loader=_dataset_loader(FIXTURE,'train',stats,[0],None,1,14,False,model_variant='idw_guided_residual')
        batch=next(iter(loader))
        self.assertTrue(torch.equal(batch['sparse'][batch['mask']==0],torch.zeros_like(batch['sparse'][batch['mask']==0])))
        for objective in ('l1_only','mse_only'):
            with self.subTest(objective=objective):
                cfg={**config,'objective_variant':objective,'output_dir':'results/fixture_objective'}
                validate_training_config(cfg)
                torch.manual_seed(14)
                g,d=_models(cfg);opt=torch.optim.Adam(g.parameters(),lr=.0002)
                before_g=[p.detach().clone() for p in g.parameters()]
                before_d=[p.detach().clone() for p in d.parameters()]
                losses=_train_epoch(g,d,loader,opt,None,100.,'idw_guided_residual',objective)
                self.assertIsNone(losses['discriminator_loss'])
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before_g,g.parameters())))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(before_d,d.parameters())))
                self.assertTrue(all(p.grad is None for p in d.parameters()))
                m1=evaluate_epoch(g,d,loader,stats,100.,'idw_guided_residual',objective)
                m2=evaluate_epoch(g,d,loader,stats,100.,'idw_guided_residual',objective)
                self.assertEqual(m1,m2)
                with torch.no_grad():_,prediction=g(batch['sparse'],batch['mask'],batch['idw'])
                physical=restore_observed(prediction,batch['mask'],stats,batch['observed_ic_physical'])
                self.assertTrue(torch.equal(physical[batch['mask'].bool()],batch['observed_ic_physical'][batch['mask'].bool()]))
                stop=NumericalBestEarlyStopping(6,.0001);stop.update(m1['hidden_rmse_physical_ic'],1)
                identity={'train_ids_sha256':'fixture'}
                payload=_checkpoint_payload(1,stop,cfg,identity,stats,g,d,opt,None)
                self.assertEqual(payload['objective_variant'],objective)
                self.assertIsNone(payload['optimizer_discriminator_state'])
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'best.pt';_atomic_checkpoint(path,payload)
                    g2,d2=_models(cfg);opt2=torch.optim.Adam(g2.parameters(),lr=.0002)
                    epoch,_=load_training_checkpoint(path,cfg,identity,stats,g2,d2,opt2,None)
                    self.assertEqual(epoch,1)
                    self.assertEqual(opt.state_dict()['state'].keys(),opt2.state_dict()['state'].keys())
                    bad={**cfg,'objective_variant':'mse_only' if objective=='l1_only' else 'l1_only'}
                    with self.assertRaisesRegex(ValueError,'mismatch'):
                        load_training_checkpoint(path,bad,identity,stats,g2,d2,opt2,None)

    def test_legacy_compatibility(self):
        cfg=json.loads((ROOT/'schemagan3d/v3_idw_guided_pilot_config.json').read_text())
        validate_training_config(cfg)
        self.assertEqual(_models(cfg)[0].enc1[0].in_channels,3)
        with self.assertRaisesRegex(ValueError,'reconstruction-only'):
            validate_training_config({**cfg,'model_variant':'baseline','objective_variant':'mse_only'})


if __name__=='__main__':unittest.main()
