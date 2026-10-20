import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from synthetic3d.v2.core import Grid
from .models import Generator3D
from .v3_adapter import (V3Normalization, V3VolumeDataset, HomogeneousShapeBatchSampler,
                         compute_train_normalization, restore_observed, xyz_to_czyx)
from .v3_eval_protocol import PROFILE_COUNTS, idw_physical
from .v3_idw_guided import (IDWGuidedVolumeDataset, IDWResidualGenerator3D,
                            idw_normalized_from_observed, make_models,
                            residual_generator_objective)
from .v3_losses import hidden_valid_l1

ROOT = Path(__file__).resolve().parents[1] / 'data/synthetic3d_v2/prototype_geotechnical_v3'
torch.set_num_threads(1)


class IDWGuidedTests(unittest.TestCase):
    def setUp(self):
        self.stats = V3Normalization(2., .5)

    def _input(self, grid, sites, values):
        observed = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
        physical = np.zeros_like(observed, dtype=np.float32)
        for (x, y), value in zip(sites, values):
            observed[x, y, :] = True
            physical[x, y, :] = value
        return torch.from_numpy(xyz_to_czyx(physical)), torch.from_numpy(xyz_to_czyx(observed.astype(np.float32)))

    def test_one_cpt_is_constant_and_hidden_values_cannot_leak(self):
        grid = Grid(12, 12, 12, 2., 3., 1.)
        physical, mask = self._input(grid, [(2, 3)], [2.25])
        first = idw_normalized_from_observed(physical, mask, grid, self.stats)
        self.assertTrue(torch.allclose(first, torch.full_like(first, .5)))
        altered = physical.clone()
        altered[mask == 0] = 100000.
        self.assertTrue(torch.equal(first, idw_normalized_from_observed(altered, mask, grid, self.stats)))

    def test_multiple_cpt_uses_physical_xy_power_two_and_exact_observations(self):
        grid = Grid(12, 12, 12, 1., 4., 1., origin_x_m=100., origin_y_m=50., azimuth_deg=23.)
        physical, mask = self._input(grid, [(0, 0), (2, 1)], [2., 3.])
        result = idw_normalized_from_observed(physical, mask, grid, self.stats)
        xyz = result[0].numpy().transpose(2, 1, 0) * self.stats.std + self.stats.mean
        point = np.array(grid.xy_m(1, 0))
        d0 = np.linalg.norm(point - np.array(grid.xy_m(0, 0)))
        d1 = np.linalg.norm(point - np.array(grid.xy_m(2, 1)))
        expected = (2. / d0**2 + 3. / d1**2) / (1. / d0**2 + 1. / d1**2)
        self.assertAlmostEqual(float(xyz[1, 0, 0]), expected, places=6)
        self.assertAlmostEqual(float(xyz[0, 0, 0]), 2., places=6)
        self.assertAlmostEqual(float(xyz[2, 1, 0]), 3., places=6)
        baseline = idw_physical(physical[0].numpy().transpose(2, 1, 0),
                                mask[0].numpy().transpose(2, 1, 0).astype(bool), grid, power=2.)
        np.testing.assert_allclose(xyz, baseline, atol=2e-7)

    def test_supported_grid_shapes_and_cpt_counts(self):
        for nx, ny, nz in ((12, 12, 12), (16, 16, 16), (24, 24, 12)):
            grid = Grid(nx, ny, nz, 2., 2., 1.)
            sites = [(x, y) for x in range(nx) for y in range(ny)][:20]
            for count in PROFILE_COUNTS:
                physical, mask = self._input(grid, sites[:count], [2. + .01 * i for i in range(count)])
                conditioned = idw_normalized_from_observed(physical, mask, grid, self.stats)
                self.assertEqual(conditioned.shape, (1, nz, ny, nx))
                self.assertTrue(torch.isfinite(conditioned).all())
            generator = IDWResidualGenerator3D(4).eval()
            sparse = torch.where(mask.bool(), (physical - self.stats.mean) / self.stats.std, 0.)
            with torch.inference_mode():
                residual, prediction = generator(sparse.unsqueeze(0), mask.unsqueeze(0), conditioned.unsqueeze(0))
            self.assertEqual(residual.shape, (1, 1, nz, ny, nx))
            self.assertEqual(prediction.shape, residual.shape)
            self.assertTrue(torch.isfinite(prediction).all())

    def test_zero_residual_is_idw_and_observed_is_exact(self):
        grid = Grid(12, 12, 12, 2., 2., 1.)
        physical, mask = self._input(grid, [(2, 3)], [2.125])
        idw = idw_normalized_from_observed(physical, mask, grid, self.stats).unsqueeze(0)
        sparse = torch.where(mask.bool(), (physical - self.stats.mean) / self.stats.std, 0.).unsqueeze(0)
        model = IDWResidualGenerator3D(4).eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        residual, prediction = model(sparse, mask.unsqueeze(0), idw)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(prediction, torch.where(mask.unsqueeze(0).bool(), sparse, idw)))
        restored = restore_observed(prediction, mask.unsqueeze(0), self.stats, physical.unsqueeze(0))
        self.assertTrue(torch.equal(restored[mask.unsqueeze(0).bool()], physical.unsqueeze(0)[mask.unsqueeze(0).bool()]))

    def test_real_adapter_wrapper_keeps_target_separate_and_batches(self):
        stats = compute_train_normalization(ROOT)
        base = V3VolumeDataset(ROOT, split='train', stats=stats)
        wrapped = IDWGuidedVolumeDataset(base)
        item = wrapped[0]
        self.assertTrue(torch.equal(item['residual_target'], item['target'] - item['idw']))
        self.assertTrue(torch.equal(item['sparse'][item['mask'] == 0], torch.zeros_like(item['sparse'][item['mask'] == 0])))
        self.assertEqual(item['idw'].shape, item['sparse'].shape)
        loader = DataLoader(wrapped, batch_sampler=HomogeneousShapeBatchSampler(wrapped, 2, group_spacing=True))
        for batch in loader:
            self.assertEqual(batch['idw'].shape, batch['target'].shape)
            self.assertEqual(batch['idw'].ndim, 5)

    def test_backward_finite_no_step_and_baseline_unchanged(self):
        torch.manual_seed(14)
        grid = Grid(12, 12, 12, 2., 2., 1.)
        physical, mask = self._input(grid, [(1, 1), (8, 8)], [2.1, 2.8])
        idw = idw_normalized_from_observed(physical, mask, grid, self.stats).unsqueeze(0)
        mask = mask.unsqueeze(0)
        sparse = torch.where(mask.bool(), (physical.unsqueeze(0) - self.stats.mean) / self.stats.std, 0.)
        target = idw + .1
        valid = torch.ones_like(mask)
        generator, discriminator = make_models(4)
        baseline = Generator3D(4)
        self.assertEqual(baseline.enc1[0].in_channels, 2)
        self.assertEqual(generator.enc1[0].in_channels, 3)
        before_g = [p.detach().clone() for p in generator.parameters()]
        before_d = [p.detach().clone() for p in discriminator.parameters()]
        before_base = [p.detach().clone() for p in baseline.parameters()]
        residual, prediction = generator(sparse, mask, idw)
        real_logits = discriminator(target, sparse, mask)
        fake_logits = discriminator(prediction, sparse, mask)
        d_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        d_loss = d_loss + torch.nn.functional.binary_cross_entropy_with_logits(fake_logits.detach(), torch.zeros_like(fake_logits))
        d_loss.backward(retain_graph=True)
        discriminator.zero_grad(set_to_none=True)
        for p in discriminator.parameters():
            p.requires_grad_(False)
        g_loss, adv, rec = residual_generator_objective(discriminator(prediction, sparse, mask), residual,
                                                         target, idw, mask, valid, reconstruction_weight=100.)
        self.assertTrue(torch.allclose(rec, hidden_valid_l1(prediction, target, mask, valid), atol=1e-6))
        g_loss.backward()
        self.assertTrue(torch.isfinite(g_loss) and torch.isfinite(d_loss) and torch.isfinite(adv) and torch.isfinite(rec))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_g, generator.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_d, discriminator.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_base, baseline.parameters())))
        with torch.inference_mode():
            self.assertEqual(baseline(sparse, mask).shape, target.shape)


if __name__ == '__main__':
    unittest.main()
