import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import Generator3D
from .v3_adapter import (V3Normalization,V3VolumeDataset,HomogeneousShapeBatchSampler,
                         assert_model_shape,compute_train_normalization,load_manifest,
                         masked_hidden_metrics,restore_observed,xyz_to_czyx)
from .v3_split import prepare_split_manifest

ROOT=Path(__file__).resolve().parents[1]/'data/synthetic3d_v2/prototype_geotechnical_v3'

class V3AdapterTests(unittest.TestCase):
    def test_real_manifest_and_one_cpt_no_leak(self):
        entries=load_manifest(ROOT);self.assertEqual(len(entries),10)
        stats=compute_train_normalization(ROOT,entries)
        ds=V3VolumeDataset(ROOT,split='train',stats=stats)
        item=ds[0];self.assertEqual(item['target'].shape,(1,16,16,16))
        self.assertEqual(item['spacing_m'].tolist(),[2.,2.,1.])
        self.assertEqual(int(item['mask'].sum().item()),16) # one full-depth CPT
        self.assertTrue(torch.all(item['sparse'][item['mask']==0]==0))
        self.assertFalse(torch.any(item['observed_ic_physical'][item['mask']==0]))
        self.assertEqual(set(item),{'sparse','mask','target','valid_mask','spacing_m','realization_id','observed_ic_physical'})
        x,y,z=2,3,4;source=np.load(ROOT/entries[0]['arrays']['Ic'])
        self.assertAlmostEqual(float(item['target'][0,z,y,x]),float((source[x,y,z]-stats.mean)/stats.std),places=6)
    def test_train_only_normalization(self):
        entries=load_manifest(ROOT);a=compute_train_normalization(ROOT,entries)
        changed=copy.deepcopy(entries)
        # Change a validation entry to point to the train array; stats must remain identical.
        val=next(e for e in changed if e['split']!='train');train=next(e for e in changed if e['split']=='train')
        val['arrays']['Ic']=train['arrays']['Ic'];val['arrays']['valid_mask']=train['arrays']['valid_mask']
        b=compute_train_normalization(ROOT,changed)
        self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stats.json';a.save(p);self.assertEqual(a,V3Normalization.load(p))
            with self.assertRaises(FileExistsError):a.save(p)
        with self.assertRaises(ValueError):V3Normalization(1.,1.,source_split='test')
        with self.assertRaisesRegex(ValueError,'provenance'):
            V3VolumeDataset(ROOT,split='validation',stats=V3Normalization(a.mean,a.std))
    def test_normalization_ignores_invalid_and_test_voxels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            train=np.full((4,4,4),999.,dtype=np.float32)
            valid=np.zeros((4,4,4),dtype=bool)
            train[0,0,0]=1.;train[0,0,1]=3.;valid[0,0,0]=valid[0,0,1]=True
            np.save(root/'train.npy',train);np.save(root/'valid.npy',valid)
            np.save(root/'test.npy',np.full((4,4,4),-10000.,dtype=np.float32))
            np.save(root/'test_valid.npy',np.ones((4,4,4),dtype=bool))
            entries=[{'split':'train','realization_id':'train-a','arrays':{'Ic':'train.npy','valid_mask':'valid.npy'}},
                     {'split':'test_id','realization_id':'test-a','arrays':{'Ic':'test.npy','valid_mask':'test_valid.npy'}}]
            stats=compute_train_normalization(root,entries)
            self.assertEqual((stats.mean,stats.std),(2.,1.))
            self.assertEqual(stats.source_train_ids,('train-a',))
    def test_axes_and_observed_restoration(self):
        a=np.fromfunction(lambda x,y,z:100*x+10*y+z,(4,4,4),dtype=int).astype(np.float32)
        t=xyz_to_czyx(a);self.assertEqual(t.shape,(1,4,4,4));self.assertEqual(t[0,3,2,1],123)
        stats=V3Normalization(2.,.5);p=torch.zeros((1,1,4,4,4));mask=torch.zeros_like(p);mask[0,0,2,1,3]=1
        measured=torch.zeros_like(p);measured[0,0,2,1,3]=2.1234567
        out=restore_observed(p,mask,stats,measured)
        self.assertEqual(out[0,0,2,1,3].item(),measured[0,0,2,1,3].item())
        valid=torch.ones_like(p);valid[0,0,0,0,0]=0
        metric=masked_hidden_metrics(out,torch.ones_like(p),mask,valid)
        self.assertEqual(metric['n'],62)
    def test_homogeneous_batches_and_shape_error(self):
        stats=compute_train_normalization(ROOT);ds=V3VolumeDataset(ROOT,split='train',stats=stats)
        loader=DataLoader(ds,batch_sampler=HomogeneousShapeBatchSampler(ds,2))
        batches=list(loader);self.assertEqual([b['target'].shape[0] for b in batches],[2,2])
        for b in batches:self.assertEqual(b['target'].shape,b['sparse'].shape)
        test=V3VolumeDataset(ROOT,split='test_parameter',stats=stats)
        loader=DataLoader(test,batch_sampler=HomogeneousShapeBatchSampler(test,2))
        self.assertEqual([tuple(b['target'].shape) for b in loader],[(1,1,12,12,12),(1,1,12,24,24)])
        with self.assertRaisesRegex(ValueError,'divisible by 4'):assert_model_shape((16,16,10))
        with self.assertRaisesRegex(ValueError,'>=8'):assert_model_shape((4,4,4))
    def test_split_six_way_and_group_integrity(self):
        records=[]
        for family in ('horizontal','inclined'):
            for i in range(3):
                g=f'{family}-{i}'
                for d in range(2):records.append(dict(realization_id=f'{g}-d{d}',lineage_group=g,family=family,geometry_tag='base',response_tag='base'))
        records += [dict(realization_id='lens-a',lineage_group='lens-pair',family='lens',geometry_tag='base',response_tag='base'),
                    dict(realization_id='lens-b',lineage_group='lens-pair',family='lens',geometry_tag='base',response_tag='base'),
                    dict(realization_id='geo',lineage_group='geo',family='horizontal',geometry_tag='high_geometry',response_tag='base'),
                    dict(realization_id='response',lineage_group='response',family='inclined',geometry_tag='base',response_tag='high_response')]
        kwargs=dict(heldout_family='lens',heldout_geometry_tag='high_geometry',heldout_response_tag='high_response')
        a=prepare_split_manifest(records,**kwargs);b=prepare_split_manifest(records,**kwargs);self.assertEqual(a,b)
        self.assertEqual(set(x['split'] for x in a['entries']),{'train','validation_id','test_id','test_family','test_geometry_parameter','test_response_parameter'})
        group={}
        for e in a['entries']:
            group.setdefault(e['lineage_group'],set()).add(e['split'])
            if e['family']=='lens':self.assertEqual(e['split'],'test_family')
        self.assertTrue(all(len(v)==1 for v in group.values()))
        bad=copy.deepcopy(records);bad.append(dict(bad[0]));bad[-1]['lineage_group']='other'
        with self.assertRaisesRegex(ValueError,'duplicate'):prepare_split_manifest(bad,**kwargs)
    def test_forward_random_weights_no_training(self):
        stats=compute_train_normalization(ROOT);ds=V3VolumeDataset(ROOT,split='train',stats=stats)
        batch=next(iter(DataLoader(ds,batch_sampler=HomogeneousShapeBatchSampler(ds,2))))
        torch.manual_seed(19);model=Generator3D(base_channels=4).eval()
        before=[p.detach().clone() for p in model.parameters()]
        with torch.inference_mode():pred=model(batch['sparse'],batch['mask'])
        self.assertEqual(pred.shape,batch['target'].shape)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(before,model.parameters())))
        physical=restore_observed(pred,batch['mask'],stats,batch['observed_ic_physical'])
        self.assertTrue(torch.equal(physical[batch['mask'].bool()],batch['observed_ic_physical'][batch['mask'].bool()]))
        self.assertGreater(masked_hidden_metrics(physical,stats.denormalize(batch['target']),batch['mask'],batch['valid_mask'])['n'],0)

if __name__=='__main__':unittest.main()
