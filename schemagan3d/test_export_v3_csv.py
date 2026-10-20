"""Small read-only-source tests for the V3 CSV exporter."""
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from .export_v3_csv import (SOURCE,COLUMNS,source_manifest,source_volume,export,validate_csv,
                            file_paths,valid_receipt)
from .v3_eval_protocol import PROFILE_COUNTS,nested_sites

class ExportV3CSVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.entries,cls.manifest_sha=source_manifest(SOURCE)

    def test_source_schema_grid_profiles_and_nested_policy(self):
        e=self.entries[0]
        arrays,observed,metadata,profiles,grid=source_volume(SOURCE,e)
        self.assertEqual(tuple(e['shape']),arrays['Ic'].shape)
        self.assertEqual(int(observed.sum()),sum(int(arrays['valid_mask'][p['grid_xy'][0],p['grid_xy'][1]].sum()) for p in profiles))
        sites=[nested_sites(grid,int(e['evaluation_site_seed']),arrays['valid_mask'],n) for n in PROFILE_COUNTS]
        self.assertTrue(all(sites[i]==sites[-1][:len(sites[i])] for i in range(len(sites))))
        self.assertEqual(COLUMNS[:13],('x','y','z','x_m','y_m','z_m','Ic','qc','fs','unit_id','sbt_id','valid_mask','observed_mask'))

    def test_exact_roundtrip_and_resume_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'export'
            result=export(SOURCE,root,self.entries[:1],self.manifest,self.manifest_sha,pilot=True)
            self.assertEqual(result['volumes'],1)
            self.assertTrue(result['round_trip_exact_all_3d_arrays'])
            csv_path,sidecar_path=file_paths(root,self.entries[0])
            self.assertTrue(valid_receipt(root,self.entries[0],self.manifest_sha))
            before=csv_path.stat().st_mtime_ns
            again=export(SOURCE,root,self.entries[:1],self.manifest,self.manifest_sha,pilot=True,resume=True)
            self.assertEqual((again['new_volumes'],again['validated_skips']),(0,1))
            self.assertEqual(csv_path.stat().st_mtime_ns,before)
            sidecar=json.loads(sidecar_path.read_text())
            arrays,observed,metadata,profiles,grid=source_volume(SOURCE,self.entries[0])
            boundary=np.asarray(sidecar['boundaries_m']['data'],dtype=np.float32)
            self.assertTrue(np.array_equal(boundary,arrays['boundaries_m']))
            self.assertEqual(sidecar['metadata'],metadata)
            self.assertEqual(sidecar['profiles'],profiles)
            self.assertEqual(validate_csv(csv_path,arrays,observed,grid),int(np.prod(arrays['Ic'].shape)))

    def test_axis_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'export'
            export(SOURCE,root,self.entries[:1],self.manifest,self.manifest_sha,pilot=True)
            csv_path,_=file_paths(root,self.entries[0])
            with csv_path.open(newline='') as stream:rows=list(csv.reader(stream))
            rows[1][0]='999'
            with csv_path.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
            self.assertFalse(valid_receipt(root,self.entries[0],self.manifest_sha))
            arrays,observed,metadata,profiles,grid=source_volume(SOURCE,self.entries[0])
            with self.assertRaisesRegex(ValueError,'axis order'):validate_csv(csv_path,arrays,observed,grid)

if __name__=='__main__':unittest.main()
