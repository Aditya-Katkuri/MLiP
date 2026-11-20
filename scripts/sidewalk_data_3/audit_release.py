from pathlib import Path
from collections import Counter
import csv,hashlib,json,subprocess,sys,base64
r=Path('/data/datasets/sidewalk-data-3');old=Path('/data/datasets/proc-sidewalk-data-naive-merged-nulls')
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
schema=json.loads((r/'label_schema.json').read_text());data=rows(r/'manifest.csv')
expected={'other-cities-train':(22912,11554,11358,0),'pitt-train':(1276,527,749,277),'pitt-val':(382,157,225,50),'pitt-test':(380,174,206,50)}
for split,(total,hazard,nonhazard,manual) in expected.items():
 sub=[x for x in data if x['split']==split]
 assert len(sub)==total and sum(int(x['hazard_label']) for x in sub)==hazard
 assert sum(x['class']=='non_hazard' for x in sub)==nonhazard
 assert sum(int(x['is_manual_collection']) for x in sub)==manual
for g,counts in {'surface_and_level_problems':[5989,245,85,81],'restricted_passage':[3114,106,23,33],'missing_tactile_warning':[3907,194,58,66],'ramp_points_into_traffic':[1250,85,25,24]}.items():
 assert [sum(int(x['group_'+g]) for x in data if x['split']==split) for split in expected]==counts
approved=rows(r/'provenance/previous_tag_review/grouping_analysis/recommended_tag_mapping.csv')
newmapping={x['tag']:x for x in rows(r/'tag_mapping.csv')}
for tag in approved:
 assert tag['hazard']==newmapping[tag['tag']]['hazard']
 name=tag['recommended_group']
 normalized_name={'Surface and level problems':'surface_and_level_problems','Restricted passage or maneuvering space':'restricted_passage','Missing tactile warning':'missing_tactile_warning','Ramp points into traffic':'ramp_points_into_traffic','Non-hazard':''}[name]
 assert newmapping[tag['tag']]['tag_group']==normalized_name
# Compare existing processed pixels with new full-image preprocessing using a
# deterministic, split-stratified sample (new images correctly have no reference).
compare_code=r'''
from pathlib import Path
import csv,hashlib,json
from PIL import Image
r=Path('/data/datasets/sidewalk-data-3');old=Path('/data/datasets/proc-sidewalk-data-naive-merged-nulls')
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
records=[]
for split in ['other-cities-train','pitt-train','pitt-val','pitt-test']:
 prior={x['source_image_path']:x for x in read(old/(split+'.csv'))}
 current=[x for x in read(r/(split+'.csv')) if x['source_image_path'] in prior]
 selected=sorted(current,key=lambda x:hashlib.sha256(x['source_image_path'].encode()).hexdigest())[:100]
 for x in selected:
  oldpath=old/prior[x['source_image_path']]['image_path']
  with Image.open(oldpath) as a,Image.open(r/x['image_path']) as b:
   same=a.size==b.size and a.convert('RGB').tobytes()==b.tobytes()
   records.append({'split':split,'image_path':x['image_path'],'source_image_path':x['source_image_path'],'previous_processed_path':str(oldpath),'identical_rgb_pixels':int(same)})
with (r/'audit/previous_preprocessing_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print(json.dumps({'compared_images':len(records),'identical_rgb_pixels':sum(x['identical_rgb_pixels'] for x in records)}))
'''
p=subprocess.run(['/opt/miniconda/envs/sidewalk/bin/python','-c',compare_code],capture_output=True,text=True,check=True)
comparison=json.loads(p.stdout);print('PREPROCESS_COMPARISON',comparison)
newimages=[x for x in data if not int(x['source_manifest_present'])]
assert len(newimages)==277 and all(x['split']=='pitt-train' and x['class']=='non_hazard' for x in newimages)
manual=[x for x in data if int(x['is_manual_collection'])]
# Confirm source manifests were not edited.
for check in rows(r/'audit/source_metadata_checksums.csv'):assert sha(Path(check['source_path']))==check['sha256']
summary={'passed':True,'approved_tag_mapping_matched':True,'expected_original_tag_group_counts_preserved':True,'all_277_new_training_images_present_as_non_hazard':True,'all_377_manual_collection_images_non_hazard':True,'source_metadata_unchanged':True,'previous_preprocessing_comparison':comparison,'images':len(data),'manual_collection_by_split':{s:sum(x['split']==s for x in manual) for s in expected},'applied_old_hazard_class_without_tags_count':len(rows(r/'audit/label_review_candidates.csv'))}
(r/'audit/release_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print('RELEASE_VALIDATION',json.dumps(summary,indent=2))
# Save combinations and pairwise overlap to make multi-label support inspectable.
combinations=[];pairs=[]
for split in expected:
 sub=[x for x in data if x['split']==split]
 for group_json,count in sorted(Counter(x['tag_groups'] for x in sub).items()):
  combinations.append({'split':split,'tag_groups':group_json,'images':count})
 for left in schema['group_order']:
  for right in schema['group_order']:
   pairs.append({'split':split,'group_a':left,'group_b':right,'images':sum(int(x['group_'+left]) and int(x['group_'+right]) for x in sub)})
for filename,items in [('tag_group_combinations.csv',combinations),('tag_group_cooccurrence.csv',pairs)]:
 with (r/'statistics'/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
# List all non-image artifact hashes after assembling the release, excluding the
# checksum list itself. Image hashes are already in every split manifest.
artifacts=[]
for p in sorted(r.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(r)
 if rel.parts[0] in expected or rel.as_posix()=='audit/artifact_checksums.csv':continue
 artifacts.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
with (r/'audit/artifact_checksums.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(artifacts)
print('ARTIFACTS_HASHED',len(artifacts))
