#!/usr/bin/env python3
"""Organize corrected pilot archives, infer ONLY B/D, and prepare blinded numbered images."""
from pathlib import Path
import argparse,csv,json,zipfile,hashlib,io,os,sys,importlib.util,platform,shutil,time,collections
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageOps,__version__ as pillow_version

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/data/datasets/sidewalk-pilot'))
parser.add_argument('--bundle',type=Path,help='Complete exported inference/ directory; defaults to root/audit/model_bundle.json')
parser.add_argument('--archive-inventory',type=Path,default=Path(__file__).resolve().parents[2]/'configs/sidewalk-pilot/archives.json')
args=parser.parse_args()
ROOT=args.root.resolve()
assert not (ROOT/'manifest.csv').exists() and not (ROOT/'review_images').exists(),'Refuse to overwrite a completed pilot build.'
BUNDLE=(args.bundle or Path(json.loads((ROOT/'audit/model_bundle.json').read_text())['inference_bundle'])).resolve()
assert BUNDLE.is_dir(),f'Missing inference bundle: {BUNDLE}'
for name in ['audit','results','source_metadata','scripts','images']:
 (ROOT/name).mkdir(parents=True,exist_ok=True)
for stretch in 'ABCD':(ROOT/'images'/stretch).mkdir(exist_ok=True)
for name in ['build_and_infer.py','evaluate_precision_at_12.py']:
 source=Path(__file__).with_name(name);target=ROOT/'scripts'/name
 if source.resolve()!=target.resolve():shutil.copy2(source,target)
if args.bundle:
 (ROOT/'audit/model_bundle.json').write_text(json.dumps({'inference_bundle':str(BUNDLE)},indent=2)+'\n')
ARCHIVES={'pitt_inf_AB.zip':set('AB'),'pitt_inf_CD.zip':set('CD')}
EXPECTED_COUNTS={'A':94,'B':96,'C':107,'D':102}
STARTED=datetime.now(timezone.utc).isoformat()

def sha(data):return hashlib.sha256(data).hexdigest()
def jsons(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def writejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def readcsv(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def writecsv(path,rows,fields=None):
 rows=list(rows)
 if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def pixsha(im):return sha(f'RGB:{im.width}x{im.height}:'.encode()+im.tobytes())

inventory=args.archive_inventory
if not inventory.is_file():inventory=ROOT/'audit/correct_archive_inventory.json'
expected_archives=json.loads(inventory.read_text())
writejson(ROOT/'audit/correct_archive_inventory.json',expected_archives)
for row in expected_archives:
 p=ROOT/'archives'/row['archive'];assert p.stat().st_size==row['bytes'] and sha(p.read_bytes())==row['sha256']
assert not (ROOT/'manifest.csv').exists() and not (ROOT/'review_images').exists(),'Refuse to overwrite a completed pilot build.'
manifest=[];stations=[];missing=[];routes={};source_records=[]
for archive_name,permitted in ARCHIVES.items():
 print('ORGANIZING',archive_name,flush=True)
 with zipfile.ZipFile(ROOT/'archives'/archive_name) as z:
  names=z.namelist();manifest_name=next(n for n in names if n.endswith('/manifest.json'));prefix=manifest_name.rsplit('/',1)[0]+'/'
  source=json.loads(z.read(manifest_name));metadata_dir=ROOT/'source_metadata'/Path(archive_name).stem;metadata_dir.mkdir()
  for n in names:
   if n.endswith('/'):continue
   rel=Path(n).relative_to(prefix.rstrip('/'));assert not rel.is_absolute() and '..' not in rel.parts
   if rel.parts[0]!='images':
    dest=metadata_dir/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
  actual_images={n for n in names if Path(n).suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}}
  referenced=set()
  for stretch in sorted(permitted):routes[stretch]=source['routes'][stretch]
  for station in source['stations']:
   stretch=station['corridor']
   if stretch not in permitted:continue # AB contains obsolete C/D placeholders; CD is authoritative for C/D.
   statbase={'stretch':stretch,'street':routes[stretch]['street'],'station_id':station['station_id'],'station_m':station['station_m'],
             'latitude':station['lat'],'longitude':station['lon'],'bearing_deg':station['bearing_deg'],'source_archive':archive_name}
   stations.append({**statbase,'station_metadata_json':jsons(station)})
   for item in station['images']:
    common={**statbase,'side':item['side'],'view_number':item['view_number'],'visibility_status':item['status'],'visibility_note':item.get('note','')}
    if not item.get('file'):
     missing.append({**common,'source_metadata_json':jsons(item)});continue
    name=prefix+item['file'];assert name in names;referenced.add(name)
    path=Path(item['file']);assert path.parts[:2]==('images',stretch)
    payload=z.read(name);target=ROOT/path;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();target.write_bytes(payload)
    with Image.open(io.BytesIO(payload)) as im:
     im.load();format_name=im.format;mode=im.mode;width,height=im.size
     rgb=ImageOps.exif_transpose(im).convert('RGB');pixel_hash=pixsha(rgb)
    image_id=Path(item['file']).stem
    row={**common,'image_id':image_id,'image_path':path.as_posix(),'source_member':name,'source_file_sha256':sha(payload),'source_rgb_pixel_sha256':pixel_hash,
         'original_filename':path.name,'source_width':width,'source_height':height,'source_mode':mode,'source_format':format_name,'source_bytes':len(payload),
         'source_image_id':item.get('source_image_id',''),'source_url':item.get('source_url',''),'creator':item.get('creator',''),'creator_url':item.get('creator_url',''),
         'captured_date':item.get('captured_date',''),'license':item.get('license',''),'license_url':item.get('license_url',''),
         'crop_source_pixel_box':jsons(item.get('crop_source_pixel_box')),'source_metadata_json':jsons(item),
         'ground_truth_hazard':'','inference_requested':int(stretch in 'BD')}
    manifest.append(row)
  assert actual_images==referenced, f'Images not represented by authorized stretches: {actual_images-referenced}'

assert dict(collections.Counter(r['stretch'] for r in manifest))==EXPECTED_COUNTS
assert len({r['image_id'] for r in manifest})==len(manifest)
manifest.sort(key=lambda r:(r['stretch'],int(r['station_m']),0 if r['side']=='left' else 1,int(r['view_number']),r['image_id']))
for stretch in 'ABCD':
 for i,row in enumerate([r for r in manifest if r['stretch']==stretch],1):row['natural_order']=i
writecsv(ROOT/'manifest.csv',manifest);writecsv(ROOT/'stations.csv',stations);writecsv(ROOT/'missing_views.csv',missing)
writejson(ROOT/'routes.json',routes)
writecsv(ROOT/'coverage.csv',[{'stretch':s,**routes[s],'image_count':EXPECTED_COUNTS[s],
    'clear_images':sum(r['stretch']==s and r['visibility_status']=='clear' for r in manifest),
    'partial_images':sum(r['stretch']==s and r['visibility_status']=='partial' for r in manifest),
    'missing_image_slots':sum(r['stretch']==s for r in missing)} for s in 'ABCD'])
# Preserve duplicates as supplied, but make image-level dependence inspectable.
byhash=collections.defaultdict(list)
for r in manifest:byhash[r['source_rgb_pixel_sha256']].append(r)
duplicates=[{'pixel_sha256':h,'group_size':len(rs),'image_id':r['image_id'],'stretch':r['stretch'],'image_path':r['image_path']} for h,rs in byhash.items() if len(rs)>1 for r in rs]
writecsv(ROOT/'audit/duplicate_images.csv',duplicates,['pixel_sha256','group_size','image_id','stretch','image_path'])

# All model files were verified on extraction; verify again before executing the saved inference code.
for path,h in json.loads((BUNDLE/'files_sha256.json').read_text()).items():assert sha((BUNDLE/path).read_bytes())==h
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER',None)
import torch,transformers
os.environ['OMP_NUM_THREADS']='8';torch.set_num_threads(8);torch.set_num_interop_threads(2)
spec=importlib.util.spec_from_file_location('pilot_saved_inference',BUNDLE/'sidewalk_inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
selected=json.loads((BUNDLE/'selection.json').read_text())
model=module.SidewalkPredictor(BUNDLE,device='cpu')
assert model.metadata['hazard']['trial_id']==selected['hazard']['hazard_deployment_trial']
assert model.metadata['tag_groups']['trial_id']==selected['tag_groups']['group_deployment_trial']
inputs=[r for r in manifest if r['stretch'] in 'BD'];assert len(inputs)==198
all_predictions=[];start=time.perf_counter()
for begin in range(0,len(inputs),16):
 block=inputs[begin:begin+16]
 returned=model.predict([ROOT/r['image_path'] for r in block],batch_size=16)
 assert len(returned)==len(block)
 for original,pred in zip(block,returned):
  assert Path(pred['image_path'])==ROOT/original['image_path']
  assert np.isfinite(pred['p_hazard']) and 0<=pred['p_hazard']<=1
  all_predictions.append({**original,**{k:v for k,v in pred.items() if k!='image_path'}})
 writecsv(ROOT/'results/B_D_predictions_unsorted.csv',all_predictions)
 print('PREDICTED',len(all_predictions),'/',len(inputs),'seconds',round(time.perf_counter()-start,1),flush=True)
assert {r['image_id'] for r in all_predictions}=={r['image_id'] for r in inputs}
assert not {r['image_id'] for r in all_predictions}&{r['image_id'] for r in manifest if r['stretch'] in 'AC'}

review=ROOT/'review_images';review.mkdir();mapping=[];review_rows=[];credits=[];ranked_by_stretch={}
for stretch in 'ABCD':
 folder=review/stretch;folder.mkdir()
 rows=[r for r in manifest if r['stretch']==stretch]
 if stretch in 'BD':rows=sorted([r for r in all_predictions if r['stretch']==stretch],key=lambda r:(-r['p_hazard'],r['image_id']))
 else:rows=sorted(rows,key=lambda r:int(r['natural_order']))
 for number,row in enumerate(rows,1):
  name=f'{number:04d}.png';rel=f'{stretch}/{name}'
  with Image.open(ROOT/row['image_path']) as im:
   rgb=ImageOps.exif_transpose(im).convert('RGB')
   # Preserve the native image content for human review; strip metadata and old filenames.
   clean=Image.frombytes('RGB',rgb.size,rgb.tobytes());clean.save(review/rel,format='PNG',compress_level=6)
  with Image.open(review/rel) as check:
   check.load();assert check.mode=='RGB' and pixsha(check)==row['source_rgb_pixel_sha256']
   assert not check.getexif() and not check.info,'Unexpected embedded metadata in blinded copy'
  entry={'stretch':stretch,'image_number':number,'review_file':rel,'image_id':row['image_id'],'original_image_path':row['image_path'],
         'natural_order':row['natural_order'],'ordering_method':'descending_p_hazard' if stretch in 'BD' else 'natural_station_side_view',
         'p_hazard':row.get('p_hazard',''),'model_rank':number if stretch in 'BD' else '',
         'source_file_sha256':row['source_file_sha256'],'source_rgb_pixel_sha256':row['source_rgb_pixel_sha256'],
         'review_file_sha256':sha((review/rel).read_bytes()),'source_archive':row['source_archive'],'source_member':row['source_member']}
  mapping.append(entry)
  review_rows.append({'stretch':stretch,'image_number':number,'image_file':rel,'hazard_label':'','notes':''})
  credits.append(f"| {rel} | [{row['creator']}]({row['creator_url']}) | {row['captured_date']} | [source]({row['source_url']}) | {row['license']} |")
  row['rank']=number;row['review_file']=rel
 if stretch in 'BD':
  ranked_by_stretch[stretch]=rows
  writecsv(ROOT/'results'/(stretch+'_ranked_predictions.csv'),rows)
  writecsv(ROOT/'results'/(stretch+'_top12.csv'),rows[:12])
writecsv(ROOT/'results/review_mapping.csv',mapping)
writecsv(review/'labels_template.csv',review_rows)
writecsv(ROOT/'results/B_D_ranked_predictions.csv',[r for s in 'BD' for r in ranked_by_stretch[s]])
writecsv(ROOT/'results/blinded_order_mapping.csv',mapping)

(review/'README.txt').write_text('''Sidewalk image review

Folders A, B, C and D each contain one stretch of sidewalk. Review the numbered
images in numeric order and record your judgments in a personal copy of
labels_template.csv. Preserve the stretch and image_number columns.

Use hazard, non_hazard, or uncertain in hazard_label. Leave a short note where
helpful. Label every image; do not infer the label from the image number. Read
LABEL_DEFINITIONS.txt before starting and work independently of other reviewers.
If the relevant area is not visible or the image is too unclear to decide, record
uncertain. An uncertain or blank entry is not counted as a non-hazard.

Images are shown at their native dimensions and contain no model predictions.
Some views may show the same physical location at different dates or from different
angles. The imagery reflects the capture dates listed in ATTRIBUTION.md.
''')
(review/'LABEL_DEFINITIONS.txt').write_text('''Hazard labeling guide

Mark hazard when the visible sidewalk shows at least one of these problems:
1. Surface or level problems: broken/very uneven surfaces, substantial bumps,
   abrupt level differences, stairs, steep/slanted transitions, pooled water,
   an uncovered manhole, or a curb ramp not level with the street.
2. Restricted passage or maneuvering space: a narrow sidewalk/passage or
   insufficient landing/turning space at a ramp.
3. Missing tactile warning: a crossing/ramp that lacks the tactile warning surface.
4. Ramp points into traffic: the ramp directs travel into traffic rather than
   aligning with the intended crossing.

Do not mark hazard solely because cracks, grass, brick paving, vegetation, a pole,
parked vehicle or another object is present. Look for one of the problems above.
These categories follow the study's labeling policy, not a comprehensive safety
assessment. Do not infer missing tactile warnings or ramp direction when a ramp
or crossing is not visible. If the visible evidence is insufficient, use uncertain.

Mark non_hazard when the visible evidence supports no listed hazard. Use uncertain
when you cannot decide. Do not use image quality or file ordering as a hazard label.
''')
(review/'ATTRIBUTION.md').write_text('# Image attribution\n\nMapillary source imagery is CC BY-SA 4.0; see [license](https://creativecommons.org/licenses/by-sa/4.0/).\nSource packages supplied cropped/resized images. This review copy applies EXIF orientation, converts to RGB PNG, and removes embedded metadata; it adds no scene pixels.\n\n| Image | Creator | Capture date | Source | License |\n|---|---|---|---|---|\n'+'\n'.join(credits)+'\n')

# Private coordinator documentation stays outside the blinded image directory.
runtime={'started_at_utc':STARTED,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'model_bundle':str(BUNDLE),
         'hazard_trial':model.metadata['hazard']['trial_id'],'hazard_config':model.metadata['hazard']['cfg'],
         'tag_group_trial':model.metadata['tag_groups']['trial_id'],'tag_group_config':model.metadata['tag_groups']['cfg'],
         'hazard_threshold':model.hazard_threshold,'group_thresholds':model.group_thresholds.tolist(),
         'python':sys.version,'torch':str(torch.__version__),'transformers':transformers.__version__,'numpy':np.__version__,'Pillow':pillow_version,
         'device':'cpu','backbone_precision':'float32','head_precision':'float32','threads':8,'batch_size':16,
         'inference_stretches':['B','D'],'inference_image_count':len(all_predictions),'image_counts':EXPECTED_COUNTS,
         'model_selection_used_pilot_labels':False,'ranking':'descending p_hazard; image_id ascending breaks exact ties',
         'rank_threshold_filter':False,'natural_order':'station_m ascending, left before right, view_number ascending, image_id ascending',
         'input_archive_hashes':{r['archive']:r['sha256'] for r in expected_archives},
         'script_sha256':sha(Path(__file__).read_bytes()),'prediction_seconds':round(time.perf_counter()-start,3)}
writejson(ROOT/'results/run_metadata.json',runtime)
for name in ['selection.json','preprocessing.json','label_schema.json','checkpoint_reload_verification.json']:
 shutil.copy2(BUNDLE/name,ROOT/'results'/('model_'+name))

private_readme='''# Sidewalk pilot: coordinator files

Use `review_images/` or `review_images.zip` for blind review. Do not share this
README, results/, source_metadata/, or the model scores until reviews are locked.

## Data and ordering

Correct source archives are archives/pitt_inf_AB.zip and archives/pitt_inf_CD.zip.
All 399 available images are included: A=94, B=96, C=107, D=102. Missing requested
views have no image and are recorded in missing_views.csv; no images are invented.
Only these two archives are accepted. Historical replacement cleanup, when present,
is recorded in audit/replaced_input_cleanup.json. AB's legacy C/D placeholders are ignored:
A/B metadata comes only from the AB archive and C/D only from the CD archive.

- A: Butler Street, from 40th Street, 0–500 m; natural station/side/view order.
- B: Penn Avenue, from 40th Street, 0–500 m; decreasing model hazard confidence.
- C: downtown Penn Avenue, from 10th Street, 0–500 m; natural station/side/view order.
- D: downtown Liberty Avenue, from 10th Street, 0–500 m; decreasing model hazard confidence.

Numbers restart at 0001 in each folder. B/0001.png and D/0001.png have the highest
hazard score in their respective stretches. A/C were never passed to the model.
All B/D images are ranked; the decision threshold does not filter the ranking.
Natural order is station distance ascending, left then right, and view number
ascending. Exact confidence ties use original image_id ascending.

The review PNGs preserve native, EXIF-oriented RGB pixels losslessly and strip
embedded metadata/original filenames. They have no confidence scores or ordering
method in the review package. Model input preprocessing separately applies the
saved full-image RGB bicubic 224×224 resize and ImageNet normalization.

## Models and results

The uploaded completed dataset-3 experiment supplies the validation-selected
best_hazard.pt and best_tag_groups.pt. No pilot labels or test results were used
for selection, calibration or retraining. The model bundle's hashes were verified.
B_D_ranked_predictions.csv and the per-stretch files in results/ contain every
probability and prediction. B_top12.csv and D_top12.csv contain the top 12 images.
review_mapping.csv joins review numbers to originals and preserves natural order.
run_metadata.json records exact model identities, preprocessing, software and inputs.
The saved standalone predictor ran on server CPU in float32; the Colab run used
T4 float16 backbone features. Small numerical differences across precision/devices
are possible. The ranking here is frozen to the saved scores.

## Evaluation after independent review

Each reviewer gets a separate copy of review_images/labels_template.csv. No model
prediction is a ground-truth label, and source visibility statuses clear/partial/
blocked are not hazard labels. No precision@12 can be measured before review.

Run `python scripts/evaluate_precision_at_12.py --labels reviewer.csv --root
/data/datasets/sidewalk-pilot --output results/reviewer_precision_at_12.csv` as a
single command. Precision@12 is hazards among images 0001–0012 divided by 12,
separately for each stretch. Unknown/blank top-12 labels produce bounds rather
than a fabricated complete score or a smaller denominator. Resolve disagreements
independently and preserve each reviewer's original labels before adjudication.

The evaluator also reports natural-order precision@12 for B/D using the same
review labels and recorded natural order. This gives a within-stretch ordering
comparison; differences between A/C and B/D alone also reflect different streets.
When all labels are resolved, overall hazard prevalence provides the expected
precision of uniformly random image selection. Images are the evaluation units;
there is no station/source-photo deduplication or diversity reranking. Several
views may show the same place. Exact duplicates are recorded in audit/duplicate_images.csv.

## Provenance

manifest.csv stores source/archive paths, hashes, city/station geometry, capture
information, visibility status, attribution and original metadata. routes.json,
stations.csv and source_metadata/ preserve geographic and collection context.
Original image bytes remain in images/, and the two input ZIPs remain in archives/.
All available metadata is retained; no original hazard labels exist for these
pilot images. audit/verification.json records the finished checks.
'''
(ROOT/'README.md').write_text(private_readme.replace('/data/datasets/sidewalk-pilot',str(ROOT)))

# Check all review copies, rank alignment, omitted A/C predictions and blank labels.
for stretch in 'ABCD':
 ms=[r for r in mapping if r['stretch']==stretch]
 assert [r['image_number'] for r in ms]==list(range(1,EXPECTED_COUNTS[stretch]+1))
 assert len(list((review/stretch).glob('*.png')))==EXPECTED_COUNTS[stretch]
 if stretch in 'BD':
  ps=[r['p_hazard'] for r in ms];assert all(a>=b for a,b in zip(ps,ps[1:]))
 else:assert [r['natural_order'] for r in ms]==list(range(1,len(ms)+1)) and all(r['p_hazard']=='' for r in ms)
assert len(review_rows)==399 and all(r['hazard_label']=='' for r in review_rows)
assert all(sha((ROOT/r['image_path']).read_bytes())==r['source_file_sha256'] for r in manifest)
assert not any('prediction' in p.name.lower() or 'score' in p.name.lower() or 'mapping' in p.name.lower() for p in review.rglob('*'))
assert not any((ROOT/'archives'/name).exists() for name in ['pitt_inference_data_1.zip','pitt_inference_data_2.zip'])
writejson(ROOT/'audit/verification.json',{'passed':True,'original_images':399,'review_images':399,'inference_images':198,
         'inference_only_B_D':True,'B_D_descending_confidence':True,'A_C_natural_order':True,
         'all_review_pixels_match_originals':True,'all_review_metadata_stripped':True,'review_labels_blank':True,
         'incorrect_archive_copies_removed':True,'exact_duplicate_groups':sum(len(rs)>1 for rs in byhash.values()),
         'precision_at_12_status':'awaiting independent human labels'})
with zipfile.ZipFile(ROOT/'review_images.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(review.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT))
writejson(ROOT/'COMPLETE.json',{'finished_at_utc':datetime.now(timezone.utc).isoformat(),'counts':EXPECTED_COUNTS,'inference_images':198,
 'review_archive_sha256':sha((ROOT/'review_images.zip').read_bytes()),'manifest_sha256':sha((ROOT/'manifest.csv').read_bytes())})
print('COMPLETE',jsons({'root':str(ROOT),'images':399,'inferred':198,'counts':EXPECTED_COUNTS,'review_zip_bytes':(ROOT/'review_images.zip').stat().st_size}),flush=True)
