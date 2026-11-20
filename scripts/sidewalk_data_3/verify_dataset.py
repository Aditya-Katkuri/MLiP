#!/usr/bin/env python3
"""Independent, read-only validation of the published manifests and image payloads.
Writes audit/independent_validation.json. Exit nonzero on any inconsistency.
"""
import argparse,csv,hashlib,json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

def load(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--workers',type=int,default=16);a=ap.parse_args();root=a.root
 schema=json.loads((root/'label_schema.json').read_text());rows=load(root/'manifest.csv');group_order=schema['group_order'];tags=schema['all_tags'];splits=['other-cities-train','pitt-train','pitt-val','pitt-test']
 assert len({r['image_path'] for r in rows})==len(rows)==len({r['sample_id'] for r in rows})
 snapshots={}
 for split in splits:
  items=load(root/(split+'.csv'));assert items==[r for r in rows if r['split']==split]
  for r in load(root/'provenance/source_metadata'/(split+'.csv')):snapshots[r['image_path']]=r
 def check(r):
  observed={t for t in tags if r[t] and float(r[t])==1}
  active=[g for g in group_order if observed&set(schema['groups'][g]['positive_tags'])]
  assert json.loads(r['tag_groups'])==active
  assert [int(r['group_'+g]) for g in group_order]==[int(g in active) for g in group_order]
  assert int(r['hazard_label'])==int(bool(active))
  assert r['class']==('hazard' if active else 'non_hazard')
  assert Path(r['image_path']).parts[:2]==(r['split'],r['class'])
  assert int(r['num_tag_groups'])==len(active)
  if int(r['is_manual_collection']):assert not active and r['class']=='non_hazard' and all(r[t]=='' for t in tags)
  if int(r['source_manifest_present']):
   original=snapshots[r['source_image_path']];assert original==json.loads(r['original_metadata_json'])
   assert all(original[t]==r[t] for t in tags)
   assert original['class']==r['original_class'] and original['split']==r['split']
  else:assert r['original_class']=='no_obstacles_manual_collection'
  p=root/r['image_path'];assert p.is_file() and p.resolve().is_relative_to(root.resolve())
  assert sha(p.read_bytes())==r['output_file_sha256']
  with Image.open(p) as im:
   im.load();assert im.size==(224,224) and im.mode=='RGB' and im.format=='PNG'
   assert sha(b'RGB:224x224:'+im.tobytes())==r['output_rgb_pixel_sha256']
  return 1
 with ThreadPoolExecutor(max_workers=a.workers) as pool:checked=sum(pool.map(check,rows))
 assert {p.relative_to(root).as_posix() for split in splits for p in (root/split).rglob('*.png')}=={r['image_path'] for r in rows}
 for r in load(root/'audit/source_metadata_checksums.csv'):
  assert sha((root/r['snapshot_path']).read_bytes())==r['sha256']
 for r in load(root/'statistics/binary_class_counts.csv'):assert int(r['images'])==sum(x['split']==r['split'] and x['class']==r['class'] for x in rows)
 for r in load(root/'statistics/tag_group_counts.csv'):assert int(r['images'])==sum(int(x['group_'+r['tag_group']]) for x in rows if x['split']==r['split'])
 for r in load(root/'statistics/tag_counts_by_split.csv'):assert int(r['images'])==sum(bool(x[r['tag']]) and float(x[r['tag']])==1 for x in rows if x['split']==r['split'])
 result={'passed':True,'images_checked':checked,'splits':dict(Counter(r['split'] for r in rows)),'source_metadata_snapshots_verified':True,'tags_match_source_manifests':True,'group_and_binary_targets_recomputed':True,'current_statistics_verified':True,'manifest_sha256':sha((root/'manifest.csv').read_bytes())}
 (root/'audit/independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
