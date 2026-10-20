import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from .train_v3 import (_models, _dataset_loader, _train_epoch, evaluate_epoch,
                       _checkpoint_payload, _atomic_checkpoint, load_training_checkpoint,
                       NumericalBestEarlyStopping, mixed_cpt_counts, validate_training_config)
from .v3_adapter import compute_train_normalization


ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'data/synthetic3d_v2/prototype_geotechnical_v3'
torch.set_num_threads(1)


class IDWTrainingIntegrationTests(unittest.TestCase):
    def test_variant_optimizer_validation_checkpoint_resume(self):
        stats=compute_train_normalization(FIXTURE)
        loader=_dataset_loader(FIXTURE,'train',stats,[0],None,1,14,False,model_variant='idw_guided_residual')
        batch=next(iter(loader))
        self.assertEqual(batch['idw'].shape,batch['target'].shape)
        self.assertTrue(torch.equal(batch['sparse'][batch['mask']==0],torch.zeros_like(batch['sparse'][batch['mask']==0])))
        g,d=_models({'model_variant':'idw_guided_residual','base_channels':4})
        self.assertEqual(g.enc1[0].in_channels,3)
        self.assertEqual(_models({'base_channels':4})[0].enc1[0].in_channels,2)
        og=torch.optim.Adam(g.parameters(),lr=.0002);od=torch.optim.Adam(d.parameters(),lr=.0002)
        before=[p.detach().clone() for p in g.parameters()]
        losses=_train_epoch(g,d,loader,og,od,100.,'idw_guided_residual')
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,g.parameters())))
        after=[p.detach().clone() for p in g.parameters()]
        metrics=evaluate_epoch(g,d,loader,stats,100.,'idw_guided_residual')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(after,g.parameters())))
        self.assertTrue(torch.isfinite(torch.tensor(metrics['hidden_rmse_physical_ic'])))
        config={'model_variant':'idw_guided_residual','base_channels':4,'train_cpt_counts':[1,2,3,5,8,12,20],
                'primary_evaluation_cpt_count':5,'training_cpt_policy':'mixed_per_epoch'}
        identity={'train_ids_sha256':'fixture'}
        stopper=NumericalBestEarlyStopping(4,.005);stopper.update(.5,1)
        payload=_checkpoint_payload(1,stopper,config,identity,stats,g,d,og,od)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.pt';_atomic_checkpoint(path,payload)
            restored_g,restored_d=_models(config)
            restored_og=torch.optim.Adam(restored_g.parameters(),lr=.0002)
            restored_od=torch.optim.Adam(restored_d.parameters(),lr=.0002)
            epoch,restored_stop=load_training_checkpoint(path,config,identity,stats,restored_g,restored_d,restored_og,restored_od)
            self.assertEqual(epoch,1);self.assertEqual(restored_stop.best_rmse,.5)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(g.parameters(),restored_g.parameters())))
            self.assertEqual(og.state_dict()['state'].keys(),restored_og.state_dict()['state'].keys())
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_training_checkpoint(path,{**config,'model_variant':'baseline'},identity,stats,*_models({'base_channels':4}),restored_og,restored_od)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_training_checkpoint(path,config,{'train_ids_sha256':'changed'},stats,restored_g,restored_d,restored_og,restored_od)

    def test_mixed_cpt_and_frozen_pilot_configuration(self):
        counts=mixed_cpt_counts(8000,[1,2,3,5,8,12,20],14,1)
        self.assertEqual(counts,mixed_cpt_counts(8000,[1,2,3,5,8,12,20],14,1))
        self.assertEqual(max(counts.count(c) for c in set(counts))-min(counts.count(c) for c in set(counts)),1)
        config=json.loads((ROOT/'schemagan3d/v3_idw_guided_pilot_config.json').read_text())
        validate_training_config(config)
        self.assertEqual(config['subset_ids_path'],'results/schemagan3d_v3_pilot3_scaled_20e/subset_ids.json')
        self.assertEqual(config['output_dir'],'results/schemagan3d_v3_idw_guided_pilot_20e')


if __name__=='__main__':unittest.main()
