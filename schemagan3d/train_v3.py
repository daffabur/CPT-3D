"""SchemaGAN V3 dry run and opt-in pilot training with deterministic monitoring."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import Generator3D,Discriminator3D
from .losses import discriminator_loss
from .v3_losses import generator_objective,hidden_valid_l1,hidden_valid_mse
from .v3_adapter import (compute_train_normalization,V3VolumeDataset,HomogeneousShapeBatchSampler,
                         restore_observed,masked_hidden_metrics)
from .prepare_v3_small_data import validate_small_manifest
from .v3_eval_protocol import PROFILE_COUNTS
from .v3_idw_guided import IDWGuidedVolumeDataset, make_models, residual_generator_objective
from .v3_cpt_gated import GatedIDWVolumeDataset, make_models as make_gated_models
from .v3_dataset_production import canonical, digest, file_sha
from .v3_split import SPLITS


def _loader(root,split,stats,batch_size,seed):
    ds=V3VolumeDataset(root,split=split,stats=stats)
    sampler=HomogeneousShapeBatchSampler(ds,batch_size,shuffle=False,seed=seed,group_spacing=True)
    return DataLoader(ds,batch_sampler=sampler)


def dry_run(config:dict,validator=validate_small_manifest):
    if config.get('mode')!='dry_run_only':raise ValueError('train_v3 only supports dry_run_only')
    if tuple(config['evaluation']['profile_counts'])!=PROFILE_COUNTS:raise ValueError('evaluation CPT-count protocol changed')
    root=Path(config['dataset_root']);split_summary=validator(root)
    stats=compute_train_normalization(root)
    train_loader=_loader(root,config['train_split'],stats,config['batch_size'],config['batch_seed'])
    validation_loader=_loader(root,config['validation_split'],stats,config['batch_size'],config['batch_seed'])
    torch.manual_seed(config['torch_seed'])
    generator=Generator3D(config['base_channels']).eval()
    discriminator=Discriminator3D(config['base_channels']).eval()
    before_g=[x.detach().clone() for x in generator.parameters()]
    before_d=[x.detach().clone() for x in discriminator.parameters()]
    batch=next(iter(train_loader))
    if not batch['valid_mask'].bool().all():raise ValueError('existing discriminator is not valid-mask aware; G0 smoke requires all voxels valid')
    sparse,mask,target=batch['sparse'],batch['mask'],batch['target']
    # D gradients only. No optimizer exists.
    with torch.no_grad():detached_prediction=generator(sparse,mask)
    d_real=discriminator(target,sparse,mask)
    d_fake=discriminator(detached_prediction,sparse,mask)
    d_loss=discriminator_loss(d_real,d_fake)
    d_loss.backward()
    d_grad=[p.grad for p in discriminator.parameters()]
    if not d_grad or any(g is None or not torch.isfinite(g).all() for g in d_grad):raise ValueError('invalid discriminator gradient')
    if any(p.grad is not None for p in generator.parameters()):raise ValueError('generator gradient leaked into discriminator pass')
    discriminator.zero_grad(set_to_none=True)
    for p in discriminator.parameters():p.requires_grad_(False)
    # G gradients only; D acts as a fixed differentiable function.
    prediction=generator(sparse,mask)
    fake_logits=discriminator(prediction,sparse,mask)
    g_loss,g_adv,g_rec=generator_objective(fake_logits,prediction,target,mask,batch['valid_mask'],reconstruction_weight=config['reconstruction_weight'])
    g_loss.backward()
    g_grad=[p.grad for p in generator.parameters()]
    if not g_grad or any(g is None or not torch.isfinite(g).all() for g in g_grad):raise ValueError('invalid generator gradient')
    if any(p.grad is not None for p in discriminator.parameters()):raise ValueError('discriminator gradient leaked into generator pass')
    for p in discriminator.parameters():p.requires_grad_(True)
    physical=restore_observed(prediction.detach(),mask,stats,batch['observed_ic_physical'])
    observed_error=float((physical[mask.bool()]-batch['observed_ic_physical'][mask.bool()]).abs().max())
    hidden=masked_hidden_metrics(physical,stats.denormalize(target),mask,batch['valid_mask'])
    if not all(torch.equal(a,b) for a,b in zip(before_g,generator.parameters())) or not all(torch.equal(a,b) for a,b in zip(before_d,discriminator.parameters())):
        raise ValueError('model weights changed during dry run')
    validation_batch=next(iter(validation_loader))
    with torch.inference_mode():val_output=generator(validation_batch['sparse'],validation_batch['mask'])
    return {'mode':'dry_run_only','optimizer_created':False,'optimizer_step':False,'checkpoint_written':False,
            'model_weights_unchanged':True,'split_summary':split_summary,
            'normalization':{'mean':stats.mean,'std':stats.std,'source_train_ids':stats.source_train_ids},
            'train_batch_shape':list(target.shape),'train_batch_spacing_m':batch['spacing_m'].tolist(),
            'validation_batch_shape':list(validation_batch['target'].shape),'validation_output_shape':list(val_output.shape),
            'generator_output_shape':list(prediction.shape),'discriminator_real_logits_shape':list(d_real.shape),
            'discriminator_fake_logits_shape':list(d_fake.shape),
            'discriminator_loss':float(d_loss.detach()),'generator_loss':float(g_loss.detach()),
            'generator_adversarial_loss':float(g_adv.detach()),'hidden_valid_l1_normalized':float(g_rec.detach()),
            'generator_gradients_finite':True,'discriminator_gradients_finite':True,
            'observed_max_error_physical':observed_error,'random_weight_hidden_metrics_do_not_interpret':hidden}


class EarlyStopping:
    """Minimize validation hidden-valid physical-Ic RMSE."""
    def __init__(self,patience:int,min_delta:float):
        if patience<1 or not np.isfinite(min_delta) or min_delta<0:raise ValueError('invalid early stopping settings')
        self.patience=patience;self.min_delta=float(min_delta)
        self.best_rmse=float('inf');self.best_epoch=0;self.bad_epochs=0

    def update(self,rmse:float,epoch:int):
        if not np.isfinite(rmse) or epoch<1:raise ValueError('invalid validation RMSE or epoch')
        improved=rmse<self.best_rmse-self.min_delta
        if improved:
            self.best_rmse=float(rmse);self.best_epoch=epoch;self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=self.patience

    def state(self):
        return {'patience':self.patience,'min_delta':self.min_delta,'best_rmse':self.best_rmse,
                'best_epoch':self.best_epoch,'bad_epochs':self.bad_epochs}

    @classmethod
    def from_state(cls,state):
        if 'early_stop_reference_rmse' in state:
            return NumericalBestEarlyStopping.from_state(state)
        obj=cls(state['patience'],state['min_delta'])
        obj.best_rmse=float(state['best_rmse']);obj.best_epoch=int(state['best_epoch']);obj.bad_epochs=int(state['bad_epochs'])
        return obj


class NumericalBestEarlyStopping(EarlyStopping):
    """Save each numerical minimum; reset patience only for a significant improvement.

    Significance is compared with the last accepted reference, not the last
    numerical minimum. This keeps the patience rule independent of checkpoint
    selection and makes resume unambiguous.
    """
    def __init__(self,patience:int,min_delta:float):
        super().__init__(patience,min_delta)
        self.early_stop_reference_rmse=float('inf')

    def update(self,rmse:float,epoch:int):
        if not np.isfinite(rmse) or epoch<1:raise ValueError('invalid validation RMSE or epoch')
        improved=rmse<self.best_rmse
        if improved:self.best_rmse=float(rmse);self.best_epoch=epoch
        if rmse<self.early_stop_reference_rmse-self.min_delta:
            self.early_stop_reference_rmse=float(rmse);self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=self.patience

    def state(self):
        result=super().state()
        result.update(best_rmse_numeric=self.best_rmse,best_epoch_numeric=self.best_epoch,
                      early_stop_reference_rmse=self.early_stop_reference_rmse)
        return result

    @classmethod
    def from_state(cls,state):
        obj=cls(state['patience'],state['min_delta'])
        obj.best_rmse=float(state['best_rmse_numeric'])
        obj.best_epoch=int(state['best_epoch_numeric'])
        obj.bad_epochs=int(state['bad_epochs'])
        obj.early_stop_reference_rmse=float(state['early_stop_reference_rmse'])
        if obj.best_rmse!=state['best_rmse'] or obj.best_epoch!=state['best_epoch']:
            raise ValueError('numeric best state mismatch')
        return obj


def balanced_indices(entries,per_cell:int,seed:int,total=None):
    """Stable equal-size family × grid subset, independent of manifest order."""
    if per_cell<1:raise ValueError('per_cell must be positive')
    groups=defaultdict(list)
    for i,e in enumerate(entries):groups[(e['family'],e.get('grid_id',e['grid_name']))].append((i,e['realization_id']))
    keys=sorted(groups)
    if total is not None and not (per_cell*len(keys)<=total<=(per_cell+1)*len(keys)):
        raise ValueError('requested total is outside balanced cell allocation')
    extra=0 if total is None else total-per_cell*len(keys)
    result=[]
    for cell,key in enumerate(keys):
        ordered=sorted(groups[key],key=lambda pair:hashlib.sha256(f'{seed}|{pair[1]}'.encode()).digest())
        quota=per_cell+int(cell<extra)
        if len(ordered)<quota:raise ValueError(f'insufficient examples for {key}')
        result.extend(i for i,_ in ordered[:quota])
    return result


def _split_metadata(root:Path,config,stats):
    manifest=json.loads((root/'manifest.json').read_text())
    entries=manifest['entries'];groups={};counts=Counter()
    for e in entries:
        gid=e['lineage_group'];split=e['split']
        if gid in groups and groups[gid]!=split:raise ValueError('lineage crosses splits')
        groups[gid]=split;counts[split]+=1
    if set(counts)!=set(SPLITS) or config['train_split']!='train' or config['validation_split']!='validation_id':
        raise ValueError('training/validation split policy changed')
    if 'expected_split_counts' in config and dict(counts)!=config['expected_split_counts']:
        raise ValueError('dataset split counts changed')
    if manifest.get('plan_sha256')!=config['expected_plan_sha256']:
        raise ValueError('frozen plan digest mismatch')
    train_ids=tuple(sorted(e['realization_id'] for e in entries if e['split']=='train'))
    if train_ids!=stats.source_train_ids:raise ValueError('normalization train IDs mismatch')
    metadata=json.loads(Path(config['normalization_metadata']).read_text())
    ids_sha=hashlib.sha256(json.dumps(train_ids,separators=(',',':')).encode()).hexdigest()
    if (metadata['plan_sha256']!=manifest['plan_sha256'] or metadata['source_split']!='train' or
            metadata['source_volume_count']!=len(train_ids) or metadata['source_train_ids_sha256']!=ids_sha or
            abs(metadata['mean']-stats.mean)>1e-12 or abs(metadata['std']-stats.std)>1e-12):
        raise ValueError('train-only normalization metadata mismatch')
    identity={'plan_sha256':manifest['plan_sha256'],'config_sha256':manifest['config_sha256'],
              'manifest_file_sha256':file_sha(root/'manifest.json'),'split_counts':dict(counts),
              'train_ids_sha256':ids_sha,'normalization':{'mean':stats.mean,'std':stats.std,'source_split':'train',
              'source_volume_count':len(train_ids),'source_train_ids_sha256':ids_sha}}
    return entries,identity


def _variant(config):
    return config.get('model_variant','baseline')


def _objective(config):
    return config.get('objective_variant','gan_l1')


def _models(config):
    if _variant(config)=='idw_guided_gated_residual':return make_gated_models(config['base_channels'])
    return make_models(config['base_channels']) if _variant(config)=='idw_guided_residual' else (Generator3D(config['base_channels']),Discriminator3D(config['base_channels']))


def _forward(generator,batch,variant):
    if variant=='idw_guided_gated_residual':return generator(batch['sparse'],batch['mask'],batch['idw'],batch['gate_features'])
    if variant=='idw_guided_residual':return generator(batch['sparse'],batch['mask'],batch['idw'])
    return None,generator(batch['sparse'],batch['mask'])


def _generator_loss(logits,residual,prediction,batch,weight,variant,objective='gan_l1'):
    if objective in ('l1_only','mse_only'):
        if variant!='idw_guided_residual':raise ValueError('reconstruction-only objective requires IDW residual model')
        loss_fn=hidden_valid_l1 if objective=='l1_only' else hidden_valid_mse
        rec=loss_fn(residual,batch['target']-batch['idw'],batch['mask'],batch['valid_mask'])
        return weight*rec,None,rec
    if variant in ('idw_guided_residual','idw_guided_gated_residual'):
        return residual_generator_objective(logits,residual,batch['target'],batch['idw'],batch['mask'],batch['valid_mask'],reconstruction_weight=weight)
    return generator_objective(logits,prediction,batch['target'],batch['mask'],batch['valid_mask'],reconstruction_weight=weight)


def _dataset_loader(root,split,stats,indices,cpt_count,batch_size,seed,shuffle,cpt_counts_by_index=None,model_variant='baseline'):
    ds=V3VolumeDataset(root,split=split,stats=stats,evaluation_cpt_count=cpt_count,indices=indices,
                       cpt_counts_by_index=cpt_counts_by_index)
    if model_variant in ('idw_guided_residual','idw_guided_gated_residual'):
        ds=IDWGuidedVolumeDataset(ds)
        if model_variant=='idw_guided_gated_residual':ds=GatedIDWVolumeDataset(ds)
    sampler=HomogeneousShapeBatchSampler(ds,batch_size,shuffle=shuffle,seed=seed,group_spacing=True)
    return DataLoader(ds,batch_sampler=sampler,num_workers=0)


def _batch_metrics_physical(prediction,batch,stats):
    physical=restore_observed(prediction,batch['mask'],stats,batch['observed_ic_physical'])
    truth=stats.denormalize(batch['target'])
    metrics=[]
    for i in range(physical.shape[0]):
        metrics.append(masked_hidden_metrics(physical[i:i+1],truth[i:i+1],
                                             batch['mask'][i:i+1],batch['valid_mask'][i:i+1]))
    return metrics


def evaluate_epoch(generator,discriminator,loader,stats,reconstruction_weight,model_variant='baseline',objective='gan_l1'):
    generator.eval();discriminator.eval()
    sums=Counter();n=0;gate_values=[]
    with torch.inference_mode():
        for batch in loader:
            sparse,mask,target=batch['sparse'],batch['mask'],batch['target']
            if not batch['valid_mask'].bool().all():raise ValueError('G0 discriminator requires fully valid voxels')
            if model_variant=='idw_guided_gated_residual':
                residual,prediction,gate,_=generator.forward_with_gate(sparse,mask,batch['idw'],batch['gate_features'])
                hidden=batch['valid_mask'].bool()&~mask.bool()
                gate_values.append(gate[hidden].cpu().numpy())
            else:
                residual,prediction=_forward(generator,batch,model_variant)
            if objective=='gan_l1':
                real_logits=discriminator(target,sparse,mask)
                fake_logits=discriminator(prediction,sparse,mask)
                d_loss=discriminator_loss(real_logits,fake_logits)
            else:fake_logits=d_loss=None
            g_loss,_,rec=_generator_loss(fake_logits,residual,prediction,batch,reconstruction_weight,model_variant,objective)
            metrics=_batch_metrics_physical(prediction,batch,stats)
            size=len(metrics);n+=size
            sums['generator_loss_total']+=float(g_loss)*size
            if d_loss is not None:sums['discriminator_loss']+=float(d_loss)*size
            sums['reconstruction_loss_normalized_ic']+=float(rec)*size
            sums['hidden_rmse_physical_ic']+=sum(m['rmse'] for m in metrics)
            sums['hidden_mae_physical_ic']+=sum(m['mae'] for m in metrics)
    if n<1:raise ValueError('empty evaluation loader')
    result={key:value/n for key,value in sums.items()}|{'volumes':n}
    if objective!='gan_l1':result['discriminator_loss']=None
    if gate_values:
        values=np.concatenate(gate_values)
        result.update(gate_hidden_valid_mean=float(values.mean()),gate_hidden_valid_std=float(values.std()),
                      gate_hidden_valid_p10=float(np.percentile(values,10)),
                      gate_hidden_valid_p50=float(np.percentile(values,50)),
                      gate_hidden_valid_p90=float(np.percentile(values,90)))
    return result


def _train_epoch(generator,discriminator,loader,opt_g,opt_d,reconstruction_weight,model_variant='baseline',objective='gan_l1'):
    generator.train();discriminator.train();sums=Counter();n=0
    for batch in loader:
        sparse,mask,target=batch['sparse'],batch['mask'],batch['target']
        if not batch['valid_mask'].bool().all():raise ValueError('G0 discriminator requires fully valid voxels')
        if objective=='gan_l1':
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():_,detached=_forward(generator,batch,model_variant)
            d_loss=discriminator_loss(discriminator(target,sparse,mask),discriminator(detached,sparse,mask))
            if not torch.isfinite(d_loss):raise FloatingPointError('nonfinite discriminator loss')
            d_loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in discriminator.parameters()):
                raise FloatingPointError('nonfinite discriminator gradient')
            opt_d.step()
            for p in discriminator.parameters():p.requires_grad_(False)
        else:d_loss=None
        opt_g.zero_grad(set_to_none=True)
        residual,prediction=_forward(generator,batch,model_variant)
        fake_logits=discriminator(prediction,sparse,mask) if objective=='gan_l1' else None
        g_loss,_,rec=_generator_loss(fake_logits,residual,prediction,batch,reconstruction_weight,model_variant,objective)
        if not torch.isfinite(g_loss) or not torch.isfinite(rec):raise FloatingPointError('nonfinite generator/reconstruction loss')
        g_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
            raise FloatingPointError('nonfinite generator gradient')
        opt_g.step()
        if any(not torch.isfinite(p).all() for p in list(generator.parameters())+list(discriminator.parameters())):
            raise FloatingPointError('nonfinite model parameter')
        for p in discriminator.parameters():p.requires_grad_(True)
        size=target.shape[0];n+=size
        sums['generator_loss_total']+=float(g_loss.detach())*size
        if d_loss is not None:sums['discriminator_loss']+=float(d_loss.detach())*size
        sums['reconstruction_loss_normalized_ic']+=float(rec.detach())*size
    result={key:value/n for key,value in sums.items()}
    if objective!='gan_l1':result['discriminator_loss']=None
    return result


def _rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def _atomic_checkpoint(path:Path,payload):
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        torch.save(payload,temp)
        with temp.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temp,path)
        try:
            fd=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        except OSError:pass  # Filesystem-dependent directory durability.
    finally:temp.unlink(missing_ok=True)


def _checkpoint_payload(epoch,stopper,config,identity,stats,generator,discriminator,opt_g,opt_d):
    return {'schema_version':'schemagan-v3-training-checkpoint-1','epoch':epoch,
            'objective_variant':_objective(config),
            'model_variant':_variant(config),'cpt_sampling_policy':{'training':config.get('training_cpt_policy','one_count_per_epoch'),'counts':config.get('train_cpt_counts'),'evaluation_count':config.get('primary_evaluation_cpt_count')},
            'best_validation_rmse_physical_ic':stopper.best_rmse,'early_stopping':stopper.state(),
            'training_configuration':config,'configuration_sha256':digest(config),'dataset_identity':identity,
            'normalization':{'mean':stats.mean,'std':stats.std,'source_split':'train',
                             'source_train_ids_sha256':identity['train_ids_sha256']},
            'generator_state':generator.state_dict(),'discriminator_state':discriminator.state_dict(),
            'optimizer_generator_state':opt_g.state_dict(),'optimizer_discriminator_state':opt_d.state_dict() if opt_d is not None else None,
            'random_state':_rng_state()}


def load_training_checkpoint(path,config,identity,stats,generator,discriminator,opt_g,opt_d):
    # Only load a trusted checkpoint produced by this local training run.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload.get('schema_version')!='schemagan-v3-training-checkpoint-1' or
            payload.get('model_variant','baseline')!=_variant(config) or
            payload.get('objective_variant','gan_l1')!=_objective(config) or
            payload.get('configuration_sha256')!=digest(config) or
            payload.get('dataset_identity')!=identity or
            payload.get('normalization',{}).get('mean')!=stats.mean or
            payload.get('normalization',{}).get('std')!=stats.std):
        raise ValueError('checkpoint config/dataset/normalization mismatch')
    generator.load_state_dict(payload['generator_state']);discriminator.load_state_dict(payload['discriminator_state'])
    opt_g.load_state_dict(payload['optimizer_generator_state'])
    if opt_d is not None:opt_d.load_state_dict(payload['optimizer_discriminator_state'])
    elif payload['optimizer_discriminator_state'] is not None:raise ValueError('checkpoint discriminator optimizer mismatch')
    _restore_rng(payload['random_state'])
    return int(payload['epoch']),EarlyStopping.from_state(payload['early_stopping'])


def validate_resume_history(output,config,identity,completed,stopper):
    history=json.loads((output/'training_history.json').read_text())
    rows=history['rows']
    with (output/'training_history.csv').open(newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    if (history.get('dataset_identity')!=identity or history.get('training_configuration')!=config or
            not rows or rows[-1]['epoch']!=completed or len(rows)!=completed or
            len(csv_rows)!=completed or [int(r['epoch']) for r in csv_rows]!=[r['epoch'] for r in rows] or
            not (output/'best.pt').is_file()):
        raise ValueError('resume history/checkpoint mismatch; no epoch will be repeated silently')
    best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    if best.get('model_variant','baseline')!=_variant(config) or best['epoch']!=stopper.best_epoch or best['best_validation_rmse_physical_ic']!=stopper.best_rmse:
        raise ValueError('best checkpoint disagrees with last checkpoint')
    if rows[-1]['stop_reason']!='':raise ValueError('completed run cannot resume with changed epoch policy')
    return rows,completed+1


HISTORY_FIELDS=('epoch','train_cpt_count','evaluation_cpt_count','generator_loss_total','discriminator_loss',
                'reconstruction_loss_normalized_ic','validation_loss_total','validation_discriminator_loss',
                'validation_reconstruction_loss_normalized_ic','train_hidden_rmse_physical_ic',
                'train_hidden_mae_physical_ic','validation_hidden_rmse_physical_ic',
                'validation_hidden_mae_physical_ic','generalization_gap_rmse_physical_ic',
                'learning_rate_generator','learning_rate_discriminator','epoch_seconds',
                'best_validation_rmse_physical_ic','best_epoch','early_stopping_bad_epochs','stop_reason')
MIXED_HISTORY_FIELDS=HISTORY_FIELDS+('training_cpt_histogram_json','early_stop_reference_rmse_physical_ic',
                                    'best_rmse_numeric_physical_ic','best_epoch_numeric')
GATED_HISTORY_FIELDS=MIXED_HISTORY_FIELDS+tuple(
    f'{split}_gate_hidden_valid_{stat}' for split in ('train','validation')
    for stat in ('mean','std','p10','p50','p90'))


def mixed_cpt_counts(n,counts,seed,epoch):
    """Balanced, reproducible per-sample CPT counts for a given epoch."""
    if n<len(counts) or epoch<1 or len(set(counts))!=len(counts):raise ValueError('invalid mixed CPT allocation')
    values=[counts[i%len(counts)] for i in range(n)]
    np.random.default_rng(np.random.SeedSequence([seed,epoch])).shuffle(values)
    return values


def write_history(output_dir,rows,config,identity):
    fields=(GATED_HISTORY_FIELDS if _variant(config)=='idw_guided_gated_residual' else
            MIXED_HISTORY_FIELDS if config.get('training_cpt_policy')=='mixed_per_epoch' else HISTORY_FIELDS)
    if any(tuple(row)!=fields for row in rows):raise ValueError('history field schema mismatch')
    csv_temp=output_dir/('training_history.csv.tmp-'+uuid.uuid4().hex)
    with csv_temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(csv_temp,output_dir/'training_history.csv')
    payload={'schema_version':'schemagan-v3-training-history-1','metric_policy':'per-volume arithmetic mean; physical Ic RMSE/MAE; fixed nested evaluation CPT sites and equal family-grid cells',
             'training_configuration':config,'dataset_identity':identity,'rows':rows}
    json_temp=output_dir/('training_history.json.tmp-'+uuid.uuid4().hex)
    with json_temp.open('w') as f:
        json.dump(payload,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(json_temp,output_dir/'training_history.json')


def render_learning_curves(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in rows]
    fig,axes=plt.subplots(3,1,figsize=(8,10),sharex=True)
    axes[0].plot(epochs,[r['train_hidden_rmse_physical_ic'] for r in rows],label='Train RMSE')
    axes[0].plot(epochs,[r['validation_hidden_rmse_physical_ic'] for r in rows],label='Validation ID RMSE')
    best=rows[-1]['best_epoch']
    axes[0].axvline(best,color='tab:green',linestyle='--',label=f'Best epoch {best}')
    axes[0].set_ylabel('Hidden-valid Ic RMSE (physical)');axes[0].legend();axes[0].grid(True)
    generator_line=axes[1].plot(epochs,[r['generator_loss_total'] for r in rows],label='Generator total',color='tab:blue')
    discriminator_axis=axes[1].twinx()
    discriminator_line=(discriminator_axis.plot(epochs,[r['discriminator_loss'] for r in rows],
                                              label='Discriminator',color='tab:orange')
                        if any(r['discriminator_loss'] is not None for r in rows) else [])
    axes[1].set_ylabel('Generator total loss (mixed)')
    discriminator_axis.set_ylabel('Discriminator loss (BCE)')
    axes[1].legend(generator_line+discriminator_line,['Generator total']+(['Discriminator'] if discriminator_line else []));axes[1].grid(True)
    axes[2].plot(epochs,[r['generalization_gap_rmse_physical_ic'] for r in rows],label='Validation − train')
    axes[2].axhline(0,color='black',linewidth=.8)
    axes[2].set_ylabel('RMSE gap (physical Ic)');axes[2].set_xlabel('Epoch');axes[2].grid(True)
    fig.tight_layout();temp=path.with_name(path.stem+'.tmp.png')
    fig.savefig(temp,dpi=140);plt.close(fig);os.replace(temp,path)


def validate_training_config(config):
    required=('dataset_root','output_dir','normalization_metadata','expected_plan_sha256',
              'train_split','validation_split','epochs','learning_rate','batch_size','base_channels',
              'reconstruction_weight','weight_decay','patience','min_delta','random_seed',
              'train_cpt_counts','evaluation_cpt_counts','primary_evaluation_cpt_count',
              'training_per_cell','evaluation_per_cell','selection_seed')
    if any(key not in config for key in required):raise ValueError('incomplete training configuration')
    if _variant(config) not in ('baseline','idw_guided_residual','idw_guided_gated_residual'):
        raise ValueError('unsupported model variant')
    if _objective(config) not in ('gan_l1','l1_only','mse_only'):
        raise ValueError('unsupported objective variant')
    if _objective(config)!='gan_l1' and _variant(config)!='idw_guided_residual':
        raise ValueError('reconstruction-only objective requires IDW residual model')
    if _variant(config) in ('idw_guided_residual','idw_guided_gated_residual'):
        if config.get('idw_power')!=2.0 or config.get('idw_coordinate_system')!='physical_xy_metres_per_depth':
            raise ValueError('IDW conditioning protocol mismatch')
        protected=['results/schemagan3d_v3_pilot3_scaled_20e']
        if _variant(config)=='idw_guided_gated_residual':
            protected.append('results/schemagan3d_v3_idw_guided_pilot_20e')
        if Path(config['output_dir']).resolve() in {Path(x).resolve() for x in protected}:
            raise ValueError('variant output cannot overwrite existing pilot checkpoints')
    if _variant(config)=='idw_guided_gated_residual':
        if (config.get('gating_feature_policy')!='observed_cpt_count_div_20_and_nearest_physical_xy_distance_div_grid_diagonal_v1' or
                config.get('gating_initial_value')!=.25):
            raise ValueError('gating feature/initialization protocol mismatch')
        if config.get('training_cpt_policy')!='mixed_per_epoch':
            raise ValueError('gated pilot requires mixed CPT training')
    if config.get('mode')!='pilot_train' or config['train_split']!='train' or config['validation_split']!='validation_id':
        raise ValueError('training mode/splits invalid')
    if (config['epochs']<1 or config['batch_size']<1 or config['base_channels']<1 or
            config['training_per_cell']<1 or config['evaluation_per_cell']<1 or
            config['evaluation_per_cell']>config['training_per_cell']):raise ValueError('invalid pilot sizes')
    for key,per_cell in (('training_total',config['training_per_cell']),
                         ('evaluation_total',config['evaluation_per_cell'])):
        if key in config and not (9*per_cell<=config[key]<=9*(per_cell+1)):
            raise ValueError(f'invalid {key}')
    if (not np.isfinite(config['learning_rate']) or config['learning_rate']<=0 or
            not np.isfinite(config['weight_decay']) or config['weight_decay']<0 or
            not np.isfinite(config['reconstruction_weight']) or config['reconstruction_weight']<=0):
        raise ValueError('invalid optimizer/loss configuration')
    EarlyStopping(config['patience'],config['min_delta'])
    if (tuple(config['evaluation_cpt_counts'])!=PROFILE_COUNTS or
            not config['train_cpt_counts'] or
            any(c not in PROFILE_COUNTS for c in config['train_cpt_counts']) or
            config['primary_evaluation_cpt_count'] not in PROFILE_COUNTS):
        raise ValueError('CPT policy changed')
    if config.get('training_cpt_policy','one_count_per_epoch') not in ('one_count_per_epoch','mixed_per_epoch'):
        raise ValueError('unsupported training CPT policy')
    if config.get('training_cpt_policy')=='mixed_per_epoch' and tuple(config['train_cpt_counts'])!=PROFILE_COUNTS:
        raise ValueError('mixed CPT policy requires all seven counts')
    if Path(config['output_dir']).resolve()==Path(config['dataset_root']).resolve():
        raise ValueError('training output cannot be dataset root')


def prepare_monitoring(config):
    validate_training_config(config)
    root=Path(config['dataset_root'])
    stats=compute_train_normalization(root)
    entries,identity=_split_metadata(root,config,stats)
    train=[e for e in entries if e['split']=='train']
    val=[e for e in entries if e['split']=='validation_id']
    train_cells={(e['family'],e.get('grid_id',e['grid_name'])) for e in train}
    val_cells={(e['family'],e.get('grid_id',e['grid_name'])) for e in val}
    if train_cells!=val_cells:raise ValueError('train/validation family-grid cells differ')
    train_indices=balanced_indices(train,config['training_per_cell'],config['selection_seed'],config.get('training_total'))
    train_eval_indices=balanced_indices(train,config['evaluation_per_cell'],config['selection_seed'],config.get('evaluation_total'))
    val_eval_indices=balanced_indices(val,config['evaluation_per_cell'],config['selection_seed'],config.get('evaluation_total'))
    if 'subset_ids_path' in config:
        saved=json.loads(Path(config['subset_ids_path']).read_text())
        if saved['dataset_identity']!=identity:raise ValueError('saved subset dataset identity mismatch')
        for name,source,selected in [('train',train,train_indices),('train_evaluation',train,train_eval_indices),
                                     ('validation_id',val,val_eval_indices)]:
            ids=[source[i]['realization_id'] for i in selected]
            if ids!=saved['ids'][name] or hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=saved['sha256'][name]:
                raise ValueError('saved subset IDs/hash mismatch')
    if 'parent_subset_ids_path' in config:
        parent=json.loads(Path(config['parent_subset_ids_path']).read_text())
        if parent['dataset_identity']!=identity:raise ValueError('parent subset dataset identity mismatch')
        for name,source,selected in [('train',train,train_indices),('train_evaluation',train,train_eval_indices),
                                     ('validation_id',val,val_eval_indices)]:
            old_ids=parent['ids'][name]
            if hashlib.sha256(json.dumps(old_ids,separators=(',',':')).encode()).hexdigest()!=parent['sha256'][name]:
                raise ValueError('parent subset ID hash mismatch')
            if not set(old_ids).issubset({source[i]['realization_id'] for i in selected}):
                raise ValueError('parent subset is not contained in scaled subset')
    if not set(train_eval_indices).issubset(train_indices):raise ValueError('train evaluation is outside training subset')
    if len(train_eval_indices)!=len(val_eval_indices):raise ValueError('unequal evaluation sample sizes')
    return root,stats,identity,train_indices,train_eval_indices,val_eval_indices


def monitoring_dry_run(config):
    """Read-only random-weight check; creates no optimizer and takes no step."""
    root,stats,identity,_,train_indices,val_indices=prepare_monitoring(config)
    count=config['primary_evaluation_cpt_count']
    # A single deterministic cell from each split keeps this check small.
    train_indices=train_indices[:1];val_indices=val_indices[:1]
    variant=_variant(config)
    train_loader=_dataset_loader(root,'train',stats,train_indices,count,1,config['selection_seed'],False,model_variant=variant)
    val_loader=_dataset_loader(root,'validation_id',stats,val_indices,count,1,config['selection_seed'],False,model_variant=variant)
    torch.manual_seed(config['random_seed'])
    g,d=_models(config)
    before_g=[p.detach().clone() for p in g.parameters()]
    before_d=[p.detach().clone() for p in d.parameters()]
    train=evaluate_epoch(g,d,train_loader,stats,config['reconstruction_weight'],variant,_objective(config))
    val=evaluate_epoch(g,d,val_loader,stats,config['reconstruction_weight'],variant,_objective(config))
    unchanged=(all(torch.equal(a,b) for a,b in zip(before_g,g.parameters())) and
               all(torch.equal(a,b) for a,b in zip(before_d,d.parameters())))
    if not unchanged:raise ValueError('dry-run model weights changed')
    return {'mode':'monitoring_dry_run','model_variant':variant,'optimizer_created':False,'optimizer_step':False,
            'checkpoint_written':False,'model_weights_unchanged':True,
            'random_weight_metrics_not_model_performance':True,
            'dataset_identity':identity,'normalization':{'mean':stats.mean,'std':stats.std,
            'source_split':'train','source_volume_count':len(stats.source_train_ids)},
            'evaluation_cpt_count':count,'train':train,'validation_id':val,
            'generalization_gap_rmse_physical_ic':val['hidden_rmse_physical_ic']-train['hidden_rmse_physical_ic']}


def train_pilot(config,resume=False):
    """Explicit opt-in training entry point. Never called by dry-run mode."""
    root,stats,identity,train_indices,train_eval_indices,val_indices=prepare_monitoring(config)
    output=Path(config['output_dir'])
    if output.exists() and not resume:raise FileExistsError('training output exists; use explicit resume')
    if resume and not (output/'last.pt').is_file():raise FileNotFoundError('last checkpoint missing')
    output.mkdir(parents=True,exist_ok=True)
    if not resume and any(output.iterdir()):raise FileExistsError('training output is not empty')
    manifest_entries=json.loads((root/'manifest.json').read_text())['entries']
    split_entries={split:[e for e in manifest_entries if e['split']==split] for split in ('train','validation_id')}
    subset_ids={name:[split_entries[split][i]['realization_id'] for i in indices]
                for name,split,indices in [('train','train',train_indices),('train_evaluation','train',train_eval_indices),
                                            ('validation_id','validation_id',val_indices)]}
    subset_payload={'schema_version':'schemagan-v3-pilot-subsets-1','selection_seed':config['selection_seed'],
                    'dataset_identity':identity,'ids':subset_ids,
                    'sha256':{key:hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
                              for key,ids in subset_ids.items()}}
    subset_path=output/'subset_ids.json'
    if resume:
        if json.loads(subset_path.read_text())!=subset_payload:raise ValueError('pilot subset changed on resume')
    else:subset_path.write_text(json.dumps(subset_payload,indent=2)+'\n')
    random.seed(config['random_seed']);np.random.seed(config['random_seed']);torch.manual_seed(config['random_seed'])
    variant=_variant(config)
    g,d=_models(config)
    opt_g=torch.optim.Adam(g.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    opt_d=(torch.optim.Adam(d.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
           if _objective(config)=='gan_l1' else None)
    mixed=config.get('training_cpt_policy')=='mixed_per_epoch'
    stopper=(NumericalBestEarlyStopping if mixed else EarlyStopping)(config['patience'],config['min_delta']);rows=[];start_epoch=1
    if resume:
        completed,stopper=load_training_checkpoint(output/'last.pt',config,identity,stats,g,d,opt_g,opt_d)
        rows,start_epoch=validate_resume_history(output,config,identity,completed,stopper)
    eval_count=config['primary_evaluation_cpt_count']
    train_eval=_dataset_loader(root,'train',stats,train_eval_indices,eval_count,config['batch_size'],
                               config['selection_seed'],False,model_variant=variant)
    val_eval=_dataset_loader(root,'validation_id',stats,val_indices,eval_count,config['batch_size'],
                             config['selection_seed'],False,model_variant=variant)
    if start_epoch>config['epochs']:raise ValueError('checkpoint already reached configured final epoch')
    for epoch in range(start_epoch,config['epochs']+1):
        started=time.perf_counter()
        train_count=('mixed' if mixed else config['train_cpt_counts'][(epoch-1)%len(config['train_cpt_counts'])])
        per_sample=(mixed_cpt_counts(len(train_indices),config['train_cpt_counts'],config['random_seed'],epoch)
                    if mixed else None)
        histogram={str(c):per_sample.count(c) for c in config['train_cpt_counts']} if mixed else None
        loader=_dataset_loader(root,'train',stats,train_indices,None if mixed else train_count,config['batch_size'],
                               config['selection_seed']+epoch,True,cpt_counts_by_index=per_sample,model_variant=variant)
        if variant=='baseline':
            losses=_train_epoch(g,d,loader,opt_g,opt_d,config['reconstruction_weight'])
            train_metrics=evaluate_epoch(g,d,train_eval,stats,config['reconstruction_weight'])
            val_metrics=evaluate_epoch(g,d,val_eval,stats,config['reconstruction_weight'])
        else:
            losses=_train_epoch(g,d,loader,opt_g,opt_d,config['reconstruction_weight'],variant,_objective(config))
            train_metrics=evaluate_epoch(g,d,train_eval,stats,config['reconstruction_weight'],variant,_objective(config))
            val_metrics=evaluate_epoch(g,d,val_eval,stats,config['reconstruction_weight'],variant,_objective(config))
        improved,stop=stopper.update(val_metrics['hidden_rmse_physical_ic'],epoch)
        reason='patience_exhausted' if stop else ('max_epochs' if epoch==config['epochs'] else '')
        row={'epoch':epoch,'train_cpt_count':train_count,'evaluation_cpt_count':eval_count,
             'generator_loss_total':losses['generator_loss_total'],
             'discriminator_loss':losses['discriminator_loss'],
             'reconstruction_loss_normalized_ic':losses['reconstruction_loss_normalized_ic'],
             'validation_loss_total':val_metrics['generator_loss_total'],
             'validation_discriminator_loss':val_metrics['discriminator_loss'],
             'validation_reconstruction_loss_normalized_ic':val_metrics['reconstruction_loss_normalized_ic'],
             'train_hidden_rmse_physical_ic':train_metrics['hidden_rmse_physical_ic'],
             'train_hidden_mae_physical_ic':train_metrics['hidden_mae_physical_ic'],
             'validation_hidden_rmse_physical_ic':val_metrics['hidden_rmse_physical_ic'],
             'validation_hidden_mae_physical_ic':val_metrics['hidden_mae_physical_ic'],
             'generalization_gap_rmse_physical_ic':val_metrics['hidden_rmse_physical_ic']-train_metrics['hidden_rmse_physical_ic'],
             'learning_rate_generator':opt_g.param_groups[0]['lr'],
             'learning_rate_discriminator':opt_d.param_groups[0]['lr'] if opt_d is not None else None,
             'epoch_seconds':time.perf_counter()-started,
             'best_validation_rmse_physical_ic':stopper.best_rmse,'best_epoch':stopper.best_epoch,
             'early_stopping_bad_epochs':stopper.bad_epochs,'stop_reason':reason}
        if mixed:
            row.update(training_cpt_histogram_json=json.dumps(histogram,sort_keys=True,separators=(',',':')),
                       early_stop_reference_rmse_physical_ic=stopper.early_stop_reference_rmse,
                       best_rmse_numeric_physical_ic=stopper.best_rmse,best_epoch_numeric=stopper.best_epoch)
        if variant=='idw_guided_gated_residual':
            row.update({f'{split}_gate_hidden_valid_{stat}':metrics[f'gate_hidden_valid_{stat}']
                        for split,metrics in (('train',train_metrics),('validation',val_metrics))
                        for stat in ('mean','std','p10','p50','p90')})
        rows.append(row);write_history(output,rows,config,identity)
        payload=_checkpoint_payload(epoch,stopper,config,identity,stats,g,d,opt_g,opt_d)
        payload['stop_reason']=reason
        if improved:_atomic_checkpoint(output/'best.pt',payload)
        _atomic_checkpoint(output/'last.pt',payload)
        render_learning_curves(rows,output/'learning_curves.png')
        if stop:break
    return {'epochs_completed':rows[-1]['epoch'],'best_epoch':stopper.best_epoch,
            'best_validation_rmse_physical_ic':stopper.best_rmse,'stop_reason':rows[-1]['stop_reason'],
            'history_csv':str(output/'training_history.csv'),'history_json':str(output/'training_history.json'),
            'learning_curves':str(output/'learning_curves.png'),'best_checkpoint':str(output/'best.pt')}


def evaluate_cpt_sweep(config,split,counts=None):
    """Separate deterministic CPT evaluation on train/validation ID only."""
    if split not in ('train','validation_id'):raise ValueError('test/holdout tuning is forbidden')
    root,stats,identity,_,train_indices,val_indices=prepare_monitoring(config)
    counts=tuple(config['evaluation_cpt_counts'] if counts is None else counts)
    if not counts or any(c not in PROFILE_COUNTS for c in counts):raise ValueError('invalid CPT sweep counts')
    path=Path(config['output_dir'])/'best.pt'
    # Trusted checkpoint produced by this run; no optimizer or step is created.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    variant=_variant(config)
    if (payload.get('schema_version')!='schemagan-v3-training-checkpoint-1' or
            payload.get('model_variant','baseline')!=variant or
            payload.get('objective_variant','gan_l1')!=_objective(config) or
            payload.get('configuration_sha256')!=digest(config) or payload.get('dataset_identity')!=identity):
        raise ValueError('best checkpoint does not match frozen config/dataset')
    g,d=_models(config)
    g.load_state_dict(payload['generator_state']);d.load_state_dict(payload['discriminator_state'])
    indices=train_indices if split=='train' else val_indices
    results={}
    for count in counts:
        loader=_dataset_loader(root,split,stats,indices,count,config['batch_size'],config['selection_seed'],False,model_variant=variant)
        results[str(count)]=evaluate_epoch(g,d,loader,stats,config['reconstruction_weight'],variant,_objective(config))
    return {'schema_version':'schemagan-v3-cpt-sweep-1','model_variant':variant,'split':split,'checkpoint_epoch':payload['epoch'],
            'dataset_identity':identity,'normalization':payload['normalization'],
            'aggregation':'arithmetic mean of per-volume metrics; physical Ic RMSE/MAE',
            'metrics_by_cpt_count':results}


def compare_cpt_sweep(config,baseline_config,split,counts=None):
    """Paired residual/baseline/IDW comparison on identical train or validation masks."""
    if _variant(config)!='idw_guided_residual' or _variant(baseline_config)!='baseline':
        raise ValueError('comparison requires IDW residual and baseline configurations')
    if split not in ('train','validation_id'):raise ValueError('test/holdout tuning is forbidden')
    root,stats,identity,_,train_indices,val_indices=prepare_monitoring(config)
    base_root,base_stats,base_identity,_,base_train,base_val=prepare_monitoring(baseline_config)
    indices=train_indices if split=='train' else val_indices
    base_indices=base_train if split=='train' else base_val
    if (root.resolve()!=base_root.resolve() or identity!=base_identity or
            stats.mean!=base_stats.mean or stats.std!=base_stats.std or indices!=base_indices or
            config['selection_seed']!=baseline_config['selection_seed']):
        raise ValueError('comparison dataset, subset, normalization, or CPT seed mismatch')
    def load_best(cfg):
        payload=torch.load(Path(cfg['output_dir'])/'best.pt',map_location='cpu',weights_only=False)
        if (payload.get('schema_version')!='schemagan-v3-training-checkpoint-1' or
                payload.get('model_variant','baseline')!=_variant(cfg) or
                payload.get('configuration_sha256')!=digest(cfg) or payload.get('dataset_identity')!=identity):
            raise ValueError('comparison checkpoint mismatch')
        generator,_=_models(cfg);generator.load_state_dict(payload['generator_state']);generator.eval()
        return generator,int(payload['epoch'])
    guided,guided_epoch=load_best(config)
    baseline,baseline_epoch=load_best(baseline_config)
    counts=tuple(config['evaluation_cpt_counts'] if counts is None else counts)
    if not counts or any(c not in PROFILE_COUNTS for c in counts):raise ValueError('invalid CPT sweep counts')
    results={}
    with torch.inference_mode():
        for count in counts:
            loader=_dataset_loader(root,split,stats,indices,count,config['batch_size'],config['selection_seed'],False,
                                   model_variant='idw_guided_residual')
            totals=defaultdict(list)
            for batch in loader:
                _,prediction=_forward(guided,batch,'idw_guided_residual')
                baseline_prediction=baseline(batch['sparse'],batch['mask'])
                for name,volume in [('idw_guided_residual',prediction),('baseline',baseline_prediction),('idw',batch['idw'])]:
                    for metric in _batch_metrics_physical(volume,batch,stats):
                        totals[name+'_rmse'].append(metric['rmse']);totals[name+'_mae'].append(metric['mae'])
            n=len(totals['baseline_rmse'])
            if n!=len(indices):raise ValueError('comparison volume count mismatch')
            results[str(count)]={'volumes':n,**{key:float(np.mean(value)) for key,value in totals.items()},
                                 'guided_minus_baseline_rmse':float(np.mean(np.array(totals['idw_guided_residual_rmse'])-np.array(totals['baseline_rmse']))),
                                 'guided_minus_idw_rmse':float(np.mean(np.array(totals['idw_guided_residual_rmse'])-np.array(totals['idw_rmse'])))}
    return {'schema_version':'schemagan-v3-paired-comparison-1','split':split,'dataset_identity':identity,
            'guided_checkpoint_epoch':guided_epoch,'baseline_checkpoint_epoch':baseline_epoch,
            'aggregation':'per-volume hidden-valid physical Ic RMSE/MAE; same batches and observed masks',
            'metrics_by_cpt_count':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('schemagan3d/v3_experiment_config.json'))
    parser.add_argument('--report',type=Path)
    parser.add_argument('--mode',choices=('existing-dry-run','monitoring-dry-run','train','cpt-sweep','comparison-sweep'),default='existing-dry-run')
    parser.add_argument('--baseline-config',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--split',choices=('train','validation_id'))
    args=parser.parse_args()
    if args.report is not None and args.report.exists():raise FileExistsError(args.report)
    config=json.loads(args.config.read_text())
    if args.mode=='existing-dry-run':
        if args.report is None:parser.error('--report required for existing dry run')
        result=dry_run(config)
    elif args.mode=='monitoring-dry-run':
        if args.resume:parser.error('--resume is only valid with --mode train')
        result=monitoring_dry_run(config)
    elif args.mode=='train':
        result=train_pilot(config,resume=args.resume)
    elif args.mode=='cpt-sweep':
        if args.split is None or args.report is None:parser.error('cpt-sweep requires --split and --report')
        if args.resume:parser.error('--resume is only valid with --mode train')
        result=evaluate_cpt_sweep(config,args.split)
    else:
        if args.split is None or args.report is None or args.baseline_config is None:
            parser.error('comparison-sweep requires --split, --report and --baseline-config')
        if args.resume:parser.error('--resume is only valid with --mode train')
        result=compare_cpt_sweep(config,json.loads(args.baseline_config.read_text()),args.split)
    if args.report is not None:args.report.write_text(json.dumps({'config':config,'result':result},indent=2)+'\n')
    print(json.dumps(result if args.mode=='train' else {'mode':args.mode,'status':'ok'}))
if __name__=='__main__':main()
