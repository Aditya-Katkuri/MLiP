#!/usr/bin/env python3
"""Build version 3 on the server; source is read-only, output publishes after validation.
Requires Pillow and matplotlib. See LABELING_RULES.md and PREPROCESSING.md in output.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageOps, __version__ as pillow_version

SPLITS = ['other-cities-train', 'pitt-train', 'pitt-val', 'pitt-test']
GROUPS = {
 'surface_and_level_problems': ['broken-surface','bumpy','uneven-surface','not-level-with-street','pooled-water','steep','surface-problem','height-difference','stairs','uneven-slanted','uncovered-manhole','very-broken'],
 'restricted_passage': ['narrow','not-enough-landing-space','narrow-sidewalk'],
 'missing_tactile_warning': ['missing-tactile-warning'],
 'ramp_points_into_traffic': ['points-into-traffic'],
}
DISPLAY = {
 'surface_and_level_problems':'Surface and level problems',
 'restricted_passage':'Restricted passage / maneuvering space',
 'missing_tactile_warning':'Missing tactile warning',
 'ramp_points_into_traffic':'Ramp points into traffic',
}
TAGS = ['brick-cobblestone','broken-surface','bumpy','level-with-sidewalk','no-pedestrian-priority','paint-fading','rail-tram-track','too-close-to-traffic','uneven-surface','very-long-crossing','missing-tactile-warning','narrow','not-enough-landing-space','not-level-with-street','parallel-lines','points-into-traffic','pooled-water','steep','surface-problem','tactile-warning','construction','fire-hydrant','garage-entrance','height-difference','litter-garbage','outdoor-dining-area','parked-bike','parked-car','parked-scooter-motorcycle','pole','sign','stairs','street-vendor','trash-recycling-can','tree','vegetation','mailbox','seating','uneven-slanted','cracks','grass','narrow-sidewalk','sand-gravel','uncovered-manhole','utility-panel','very-broken']
HAZARD_TAGS = set(t for ts in GROUPS.values() for t in ts)
EXTENSIONS = {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
SIZE = 224
RULE_VERSION = 'sidewalk-data-3-tag-groups-v1'

def now(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()
def filehash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def pixelhash(im): return digest((f'RGB:{im.width}x{im.height}:').encode()+im.tobytes())
def js(value): return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def writejson(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def writecsv(path,rows,fields=None):
 rows=list(rows)
 if fields is None:
  fields=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise'); w.writeheader();w.writerows(rows)
def readcsv(path):
 with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def tag_positive(value):
 if value == '': return False
 if float(value) not in (0,1): raise ValueError(f'Unexpected tag value {value!r}')
 return float(value)==1

def source_inventory(source):
 records=[]; ignored=[];coverage=[]
 for split in SPLITS:
  manifest=source/(split+'.csv'); rows=readcsv(manifest)
  if rows and set(rows[0]) != {'filename','class','image_path','split','normalized_x','normalized_y',*TAGS}:
   raise ValueError(f'Unknown source schema: {manifest}')
  bypath={r['image_path']:(i+2,r) for i,r in enumerate(rows)}
  assert len(bypath)==len(rows),f'Duplicate paths in {manifest}'
  files=sorted(p for p in (source/split).rglob('*') if p.is_file())
  actual=set();nnew=0
  for p in files:
   rel=p.relative_to(source).as_posix()
   if p.suffix.lower() not in EXTENSIONS:
    ignored.append({'source_image_path':rel,'bytes':p.stat().st_size,'reason':'not_an_image_extension'});continue
   assert p.resolve().is_relative_to(source.resolve()),f'Outside source: {p}'
   actual.add(rel);parts=Path(rel).parts; original_class=parts[1]
   if rel in bypath:
    line,row=bypath[rel]
    assert row['split']==split and row['class']==original_class and row['filename']==p.name
    origin='source_split_csv'
   else:
    # Only the explicitly authorized hand-collected null class may lack a manifest.
    assert split.startswith('pitt-') and original_class=='no_obstacles_manual_collection',f'Unannotated unexpected image: {rel}'
    line='';row={'filename':p.name,'class':original_class,'image_path':rel,'split':split,'normalized_x':'','normalized_y':'',**{t:'' for t in TAGS}}
    origin='discovered_manual_collection';nnew+=1
   records.append({'split':split,'rel':rel,'row':row,'line':line,'origin':origin,'stat_size':p.stat().st_size,'stat_mtime_ns':p.stat().st_mtime_ns})
  assert set(bypath)<=actual, f'Missing source images: {set(bypath)-actual}'
  coverage.append({'split':split,'source_csv_rows':len(rows),'source_image_files':len(actual),'added_images_missing_from_source_csv':nnew})
 return records,ignored,coverage

def load_provenance(source):
 bypath=defaultdict(list); metadata_files=[]
 for p in sorted(source.rglob('*.csv')):
  rel=p.relative_to(source).as_posix()
  if rel in [s+'.csv' for s in SPLITS]: continue
  metadata_files.append(p)
  for i,row in enumerate(readcsv(p),2):
   key=row.get('image_path','')
   if p.name=='manual_class_labels.csv': key=(p.parent.relative_to(source)/row['filename']).as_posix()
   if key:
    bypath[key].append({'metadata_file':rel,'csv_line':i,'record':row})
 return bypath,metadata_files

def process_one(rec,source,stage,provenance):
 p=source/rec['rel'];row=rec['row'];data=p.read_bytes()
 assert len(data)==rec['stat_size'] and p.stat().st_mtime_ns==rec['stat_mtime_ns'],f'Source changed: {p}'
 with Image.open(io.BytesIO(data)) as im:
  im.load();w,h=im.size;fmt=im.format;mode=im.mode;orientation=im.getexif().get(274,1)
  assert getattr(im,'n_frames',1)==1,f'Unexpected animated image: {p}'
  alpha_min=im.getchannel('A').getextrema()[0] if 'A' in im.getbands() else 255
  rgb=ImageOps.exif_transpose(im).convert('RGB')
  sw,sh=rgb.size;src_pixel_hash=pixelhash(rgb)
  output=rgb.resize((SIZE,SIZE),Image.Resampling.BICUBIC)
  buf=io.BytesIO();output.save(buf,format='PNG',compress_level=6);out_bytes=buf.getvalue()
 manual=row['class']=='no_obstacles_manual_collection'
 positives=[t for t in TAGS if tag_positive(row[t])]
 selected=[g for g,tags in GROUPS.items() if any(t in positives for t in tags)]
 if manual:
  assert not positives, f'Unexpected tags on manual null: {p}'
  selected=[]
 hazard=int(bool(selected));label='hazard' if hazard else 'non_hazard'
 sample_id=digest(rec['rel'].encode())[:20]
 stem=re.sub(r'[^a-zA-Z0-9._-]+','_',p.stem)[:110]
 outname=sample_id+'__'+stem+'.png';outrel=f'{rec["split"]}/{label}/{outname}'
 target=stage/outrel;target.write_bytes(out_bytes)
 tag_observed=sum(row[t]!='' for t in TAGS)
 refs=provenance.get(rec['rel'],[])
 applied=[ref['record'] for ref in refs if ref['metadata_file']=='manual_relabeling.csv']
 latest=applied[0] if applied else {}
 flags=[]
 if not tag_observed and not manual: flags.append('no_tag_annotations_not_manual_collection')
 if not tag_observed and row['class'] in ('Obstacle','SurfaceProblem'):flags.append('original_hazard_class_without_tags_assigned_non_hazard')
 if orientation not in (None,1): flags.append('exif_orientation_corrected')
 if alpha_min<255: flags.append('source_has_transparent_pixels_alpha_dropped')
 match=re.match(r'^(gsv|mapillary)-(.+?)-(\d+)-',p.name)
 city=match.group(2) if match else ('pittsburgh' if rec['split'].startswith('pitt-') else '')
 assert city,f'Unknown city: {p}'
 assert (city=='pittsburgh')==rec['split'].startswith('pitt-'),f'City/split mismatch: {p}'
 basis='manual_collection_ordinary_sidewalk' if manual else ('positive_hazard_tag' if hazard else ('no_positive_hazard_tag' if tag_observed else 'no_tag_annotations_assumed_non_hazard'))
 result={
  'sample_id':sample_id,'filename':outname,'class':label,'hazard_label':hazard,'image_path':outrel,'split':rec['split'],
  **{'group_'+g:int(g in selected) for g in GROUPS},'tag_groups':js(selected),'num_tag_groups':len(selected),'label_rule_version':RULE_VERSION,'label_basis':basis,
  'original_class':row['class'],'original_filename':p.name,'source_image_path':rec['rel'],'source_path':str(p),'source_dataset':str(source),
  'source_manifest':rec['split']+'.csv' if rec['line'] else '', 'source_manifest_csv_line':rec['line'],'source_manifest_present':int(bool(rec['line'])),'metadata_origin':rec['origin'],
  'is_manual_collection':int(manual),'collection_batch':Path(rec['rel']).parts[2] if manual and len(Path(rec['rel']).parts)>3 else '',
  'city':city,'city_basis':'source_filename' if match else 'user_supplied_pittsburgh_collection','source_label_id':match.group(3) if match else '',
  'pre_review_class':latest.get('source_class',''),'pre_review_image_path':latest.get('source_image_path',''),'applied_manual_class':latest.get('manual_label',''),'manual_label_updated_at':latest.get('manual_label_updated_at',''),
  'positive_tags':js(positives),'positive_hazard_tags':js([t for t in positives if t in HAZARD_TAGS]),'num_observed_tag_fields':tag_observed,
  'annotation_status':'manual_collection_no_tag_annotations' if manual else ('no_tag_annotations' if not tag_observed else ('positive_tags' if positives else 'annotated_no_positive_tags')),
  'normalized_x':row['normalized_x'],'normalized_y':row['normalized_y'],**{t:row[t] for t in TAGS},
  'source_file_sha256':digest(data),'source_rgb_pixel_sha256':src_pixel_hash,'source_bytes':len(data),'source_mtime_ns':rec['stat_mtime_ns'],
  'source_width':w,'source_height':h,'source_mode':mode,'source_format':fmt,'source_exif_orientation':orientation,'source_alpha_min':alpha_min,
  'oriented_width':sw,'oriented_height':sh,'crop_left':0,'crop_top':0,'crop_right':sw,'crop_bottom':sh,'preprocess_policy':'exif_transpose_rgb_full_image_bicubic_224_png_v1',
  'output_width':SIZE,'output_height':SIZE,'output_mode':'RGB','output_format':'PNG','output_bytes':len(out_bytes),'output_file_sha256':digest(out_bytes),'output_rgb_pixel_sha256':pixelhash(output),
  'quality_flags':js(flags),'original_metadata_json':js(row if rec['line'] else {}),'source_provenance_json':js(refs),
 }
 return result

def duplicate_audit(rows,stage):
 report=[];cross=[];stats={}
 for name,col in [('source_bytes','source_file_sha256'),('source_rgb_pixels','source_rgb_pixel_sha256'),('processed_rgb_pixels','output_rgb_pixel_sha256')]:
  groups=defaultdict(list)
  for r in rows:groups[r[col]].append(r)
  counts=Counter()
  for h,rs in groups.items():
   if len(rs)<2:continue
   splits=sorted({r['split'] for r in rs});multi=len(splits)>1;counts['groups']+=1;counts['images']+=len(rs)
   if multi:counts['cross_split_groups']+=1;counts['cross_split_images']+=len(rs)
   for r in rs:
    item={'hash_type':name,'sha256':h,'group_size':len(rs),'splits':js(splits),'cross_split':int(multi),'label_conflict':int(len({r['class'] for r in rs})>1),'group_label_conflict':int(len({r['tag_groups'] for r in rs})>1),'sample_id':r['sample_id'],'split':r['split'],'class':r['class'],'tag_groups':r['tag_groups'],'image_path':r['image_path'],'source_image_path':r['source_image_path']}
    report.append(item)
    if multi:cross.append(item)
  stats[name]=dict(counts)
 fields=['hash_type','sha256','group_size','splits','cross_split','label_conflict','group_label_conflict','sample_id','split','class','tag_groups','image_path','source_image_path']
 writecsv(stage/'audit/duplicate_images.csv',report,fields)
 writecsv(stage/'audit/cross_split_duplicates.csv',cross,fields)
 return stats

def validate(rows,records,source,stage,workers):
 assert len(rows)==len(records)
 assert len({r['sample_id'] for r in rows})==len(rows)
 assert len({r['image_path'] for r in rows})==len(rows)
 assert {r['source_image_path'] for r in rows}=={r['rel'] for r in records}
 def check(r):
  data=(stage/r['image_path']).read_bytes();assert digest(data)==r['output_file_sha256']
  with Image.open(io.BytesIO(data)) as im:
   im.load();assert im.size==(SIZE,SIZE) and im.mode=='RGB' and im.format=='PNG'
   assert pixelhash(im)==r['output_rgb_pixel_sha256']
  positives=[t for t in TAGS if tag_positive(r[t])]
  groups=[g for g,tags in GROUPS.items() if set(tags)&set(positives)]
  assert groups==json.loads(r['tag_groups'])
  assert r['hazard_label']==int(bool(groups))
  assert r['class']==('hazard' if groups else 'non_hazard')
  assert all(r['group_'+g]==int(g in groups) for g in GROUPS)
  if r['is_manual_collection']: assert r['hazard_label']==0
  p=source/r['source_image_path'];assert p.stat().st_size==r['source_bytes'] and p.stat().st_mtime_ns==r['source_mtime_ns']
  for t in TAGS: assert r[t]==json.loads(r['original_metadata_json']).get(t,'')
  return 1
 with ThreadPoolExecutor(max_workers=workers) as pool: checked=sum(pool.map(check,rows))
 actual={p.relative_to(stage).as_posix() for split in SPLITS for p in (stage/split).rglob('*.png')}
 assert actual=={r['image_path'] for r in rows}
 return {'passed':True,'images_checked':checked,'full_decode_all_outputs':True,'output_hashes_checked':True,'source_stat_unchanged':True,'all_source_images_represented_exactly_once':True,'tag_values_preserved':True,'binary_and_group_labels_recomputed':True,'all_manual_collection_images_non_hazard':True,'split_membership_preserved':True,'output_shape':[224,224,3],'output_format':'PNG'}

def summarize(rows,stage):
 binary=[];groupcounts=[];tagcounts=[];original=[];overlap=[];cities=[];transitions=[]
 for split in SPLITS:
  rs=[r for r in rows if r['split']==split];n=len(rs)
  for label in ['hazard','non_hazard']:
   count=sum(r['class']==label for r in rs);binary.append({'split':split,'class':label,'images':count,'percent':round(count/n*100,6),'split_total':n})
  for group in GROUPS:groupcounts.append({'split':split,'tag_group':group,'display_name':DISPLAY[group],'images':sum(r['group_'+group] for r in rs),'split_total':n})
  for tag in TAGS:tagcounts.append({'split':split,'tag':tag,'hazard_tag':int(tag in HAZARD_TAGS),'tag_group':next((g for g,ts in GROUPS.items() if tag in ts),''),'images':sum(tag_positive(r[tag]) for r in rs),'annotated_images':sum(r[tag]!='' for r in rs)})
  for cls,count in sorted(Counter(r['original_class'] for r in rs).items()):original.append({'split':split,'original_class':cls,'images':count})
  for city,count in sorted(Counter(r['city'] for r in rs).items()):cities.append({'split':split,'city':city,'images':count})
  for (cls,label),count in sorted(Counter((r['original_class'],r['class']) for r in rs).items()):transitions.append({'split':split,'original_class':cls,'class':label,'images':count})
  overlap.append({'split':split,'total_images':n,'hazard_images':sum(r['hazard_label'] for r in rs),'non_hazard_images':sum(1-r['hazard_label'] for r in rs),'images_with_multiple_tag_groups':sum(r['num_tag_groups']>1 for r in rs),'total_group_memberships':sum(r['num_tag_groups'] for r in rs),'manual_collection_images':sum(r['is_manual_collection'] for r in rs),'images_without_source_manifest':sum(not r['source_manifest_present'] for r in rs),'images_without_tag_annotations_not_manual_collection':sum(r['annotation_status']=='no_tag_annotations' for r in rs)})
 for filename,data in [('binary_class_counts.csv',binary),('tag_group_counts.csv',groupcounts),('tag_counts_by_split.csv',tagcounts),('original_class_counts.csv',original),('city_counts.csv',cities),('label_transition_counts.csv',transitions),('tag_group_overlap.csv',overlap)]:writecsv(stage/'statistics'/filename,data)
 return {'binary':binary,'groups':groupcounts,'tags':tagcounts,'overlap':overlap}

def plots(stats,stage):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Patch
 out=stage/'histograms';out.mkdir()
 plot_splits=['pitt-train','pitt-val','pitt-test','other-cities-train']
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 def save(fig,name):
  for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=170,bbox_inches='tight')
  plt.close(fig)
 for percent in [False,True]:
  fig,axs=plt.subplots(2,2,figsize=(12,8))
  for ax,split in zip(axs.flat,plot_splits):
   rs=[r for r in stats['binary'] if r['split']==split];ys=[r['percent'] if percent else r['images'] for r in rs]
   bars=ax.bar(['Hazard','Non-hazard'],ys,color=['#d96826','#357ca5'])
   ax.bar_label(bars,labels=[f'{r["percent"]:.1f}%' if percent else f'{r["images"]:,}' for r in rs],padding=3)
   ax.set_title(f'{split} (n={rs[0]["split_total"]:,})');ax.set_ylabel('Images (%)' if percent else 'Images');ax.set_ylim(0,100 if percent else max(ys)*1.19)
  fig.suptitle('Sidewalk-data-3: hazard / non-hazard');fig.tight_layout();save(fig,'binary_class_percent' if percent else 'binary_class_counts')
 fig,axs=plt.subplots(2,2,figsize=(15,9))
 labels=['Surface / level','Restricted passage','Missing tactile warning','Ramp into traffic']
 for ax,split in zip(axs.flat,plot_splits):
  rs=[r for r in stats['groups'] if r['split']==split];ys=[r['images'] for r in rs]
  bars=ax.barh(labels[::-1],ys[::-1],color=['#427eaa','#6b9e67','#a880b7','#d38b42'][::-1]);ax.bar_label(bars,padding=4);ax.set_xlim(0,max(ys)*1.18);ax.set_title(split);ax.set_xlabel('Images (groups can overlap)')
 fig.suptitle('Sidewalk-data-3: four hazard groups');fig.tight_layout();save(fig,'tag_group_counts')
 totals=Counter()
 for r in stats['tags']:totals[r['tag']]+=r['images']
 order=sorted(TAGS,key=lambda t:(-totals[t],t));colors=['#d96826' if t in HAZARD_TAGS else '#357ca5' for t in order]
 fig,axs=plt.subplots(1,4,figsize=(19,15),sharey=True)
 for ax,split in zip(axs,plot_splits):
  bytag={r['tag']:r['images'] for r in stats['tags'] if r['split']==split};ys=[bytag[t] for t in order]
  bars=ax.barh(range(len(order)),ys,color=colors);ax.bar_label(bars,fontsize=7,padding=2);ax.set_title(split);ax.set_xlim(0,max(ys)*1.2);ax.set_xlabel('Images with tag');ax.set_yticks(range(len(order)),order);ax.tick_params(axis='y',labelsize=9)
 axs[0].invert_yaxis();fig.suptitle('Sidewalk-data-3: individual tags (all 46 tags, including zero counts)')
 fig.legend(handles=[Patch(color='#d96826',label='Hazard tag'),Patch(color='#357ca5',label='Non-hazard / context-needed tag')],loc='lower center',ncol=2,bbox_to_anchor=(.55,.045))
 fig.text(.55,.018,'Images can have several tags. Untagged images, including all hand-collected nulls, do not contribute to tag bars.',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.085,1,.97));save(fig,'tag_histograms')
 for split in plot_splits:
  fig,ax=plt.subplots(figsize=(10,14));bytag={r['tag']:r['images'] for r in stats['tags'] if r['split']==split}
  vals=[bytag[t] for t in order];bars=ax.barh(order,vals,color=colors);ax.invert_yaxis();ax.bar_label(bars,padding=3,fontsize=8);ax.set_xlim(0,max(vals)*1.17);ax.set_xlabel('Images with tag (multiple tags per image possible)');ax.set_title(f'{split}: tag distribution');fig.tight_layout();save(fig,split+'_tag_histogram')
 return matplotlib.__version__

def docs(stage,source,target,rows,stats,validation,duplicates,started):
 group_lines='\n'.join(f'- **{DISPLAY[g]}** (`{g}`): '+', '.join('`'+t+'`' for t in tags)+'.' for g,tags in GROUPS.items())
 (stage/'LABELING_RULES.md').write_text(f'''# Labeling rules: {RULE_VERSION}

Model 1 predicts binary `hazard_label`: 1 = hazard, 0 = non_hazard.
Model 2 predicts four independent tag-group memberships, in the order below.
These are multi-label targets, not mutually exclusive classes. Use four sigmoid
outputs / binary cross-entropy, not a four-way softmax. Train/evaluate the second
stage on hazard images for hazard-subtype classification; a joint model may use
all rows with the explicitly stored all-zero vectors on non-hazards. Report which
population is evaluated. Group-level metrics must not assume one true group.

{group_lines}

A group is positive if ANY of its constituent tags is positive. An image is
hazardous if ANY group is positive. Context-needed and non-hazard tags remain
non-hazard, as requested; they do not cancel a co-occurring positive hazard tag.
The complete 46-tag mapping is `tag_mapping.csv` and `label_schema.json`.

All `no_obstacles_manual_collection` images are non-hazard because the user
identified them as ordinary sidewalks. This includes the 277 newly discovered
Pittsburgh training screenshots and the existing 50 validation + 50 test images.
Their original tags remain blank, not invented zero annotations.

Images with no positive hazard tags are non-hazard under this operational rule,
including images with no tag annotations at all. Absence of annotations is not
independent verification of safety. Source class names and old manual class
reviews do not override this tag rule. In particular, tagless images previously
reviewed as Obstacle/SurfaceProblem now receive non-hazard; they are explicitly
listed in `audit/label_review_candidates.csv`, with the old decisions preserved.
No new tags are inferred from original class names, filenames, or image content.

Existing splits and all image records are retained, including exact duplicates;
see `audit/cross_split_duplicates.csv` before interpreting split-independent
performance. No image is duplicated on disk merely because it has several groups.
The non-hazard outcome has the four-vector [0, 0, 0, 0].
''')
 (stage/'PREPROCESSING.md').write_text('''# Image preprocessing

All source images are read directly from sidewalk-data-2, including recursively
nested new training collections. Each image is decoded fully, EXIF-transposed
into display orientation, converted using Pillow `.convert("RGB")`, resized
from the complete image to 224 x 224 with BICUBIC interpolation, and encoded as
lossless RGB PNG (compression level 6). Alpha is dropped in the same RGB conversion
used by the previous pipeline; source alpha minima are recorded for auditing.
The full oriented image is the crop rectangle. No center crop, padding, annotation
marker removal, color enhancement, or random augmentation is applied. Aspect ratios
are resized directly to a square, matching the previous naive-merged-nulls setup.
Screenshots already showing a selected sidewalk scene receive no further crop.
This standardizes format/shape, not the original field of view of heterogeneous
sources (GSV frames, existing crops, and screenshots).

RGB uint8 pixels are stored without model-specific normalization. Training loaders
should convert to tensors and apply the selected model's normalization consistently;
do not perform another center crop or change evaluation geometry unintentionally.
Original normalized_x/y annotations are retained in SOURCE-image coordinates.
Source dimensions, EXIF orientation, full-image crop bounds and output dimensions
are recorded, so coordinates can be transformed explicitly if necessary.

Every output is reopened and decoded, its shape/mode/format and hashes are checked,
and all targets are recomputed from preserved tag values. Images use a deterministic
20-hex-character SHA-256 prefix of their source-relative path plus the source stem;
original names and paths are retained verbatim in metadata. This avoids collisions
between collection folders or source formats.
''')
 summary='| Split | Hazard | Non-hazard | Total |\n|---|---:|---:|---:|\n'
 for r in stats['overlap']:summary+=f'| {r["split"]} | {r["hazard_images"]:,} | {r["non_hazard_images"]:,} | {r["total_images"]:,} |\n'
 gr='| Tag group | Other-cities train | Pitt train | Pitt val | Pitt test |\n|---|---:|---:|---:|---:|\n'
 for g in GROUPS:
  vals=[next(r['images'] for r in stats['groups'] if r['split']==s and r['tag_group']==g) for s in SPLITS]
  gr+='| '+DISPLAY[g]+' | '+' | '.join(f'{n:,}' for n in vals)+' |\n'
 nreview=sum('original_hazard_class_without_tags_assigned_non_hazard' in r['quality_flags'] for r in rows)
 (stage/'README.md').write_text(f'''# Sidewalk dataset 3

Built on the server from `{source}`. {len(rows):,} image records, including the
277 training screenshots missing from the source CSVs. Source data is unchanged.
Existing city splits are preserved: Pittsburgh only in `pitt-*`; all other cities
in `other-cities-train`. All images are 224 x 224 RGB PNG.

{summary}
{gr}
Group counts overlap. Multiple-group images in Pitt train/val/test:
{', '.join(str(next(r['images_with_multiple_tag_groups'] for r in stats['overlap'] if r['split']==s)) for s in SPLITS[1:])}.
All 377 hand-collected null images are non-hazard (277 train, 50 val, 50 test).

## Using the dataset

Images: `<split>/hazard/*.png` and `<split>/non_hazard/*.png`.
`<split>.csv` and `manifest.csv` contain one row per image. Paths are relative to
this directory; `class` is the current binary class and `hazard_label` is 0/1.
For the second model use the four `group_*` columns, in the order in
`label_schema.json`; `tag_groups` also stores the positive names as a JSON list.
Do not use original feature classes as binary ground truth or force one group
on images with several labels. See `LABELING_RULES.md` for the exact mapping.

## Metadata and documentation

- Original classes, filenames, paths, raw 46 tag fields, coordinates, prior reviews,
  collection batch/city, source-CSV membership and label basis are in each row.
- File/pixel SHA-256 hashes, input/output dimensions, modes, formats, EXIF orientation
  and crop rectangles allow tracing and checking preprocessing.
- `DATA_DICTIONARY.md`, `LABELING_RULES.md`, `PREPROCESSING.md`, `label_schema.json`,
  and `tag_mapping.csv` define the data contract.
- `statistics/` contains binary/group/tag counts, original-class transitions,
  city distributions and overlap counts. `histograms/` contains PNG/PDF/SVG plots.
- `provenance/source_metadata/` snapshots the source CSVs, review files, README,
  split configuration, and prior processed configuration. Original row metadata
  is also embedded in `original_metadata_json` / `source_provenance_json`.
- `audit/` records newly added images, missing tags, label-review candidates,
  duplicates across and within splits, input metadata hashes and verification.
- `scripts/build_dataset.py` reproduces the build, and `scripts/verify_dataset.py`
  independently validates it. `build_config.json` records software and settings.

There are {nreview} tagless records whose old class was Obstacle or SurfaceProblem;
they are non-hazard under the agreed tag-only rule and flagged for optional review.
No deduplication or split reassignment was applied. Duplicate counts and cross-split
memberships are provided rather than silently changing the existing dataset.
Historical tag plots, if supplied, are under `provenance/previous_tag_review/`;
use the current `statistics/` and `histograms/` for this version.
''')
 (stage/'DATA_DICTIONARY.md').write_text('''# Manifest fields

All manifests use the same header. Numeric group/binary targets contain 0 or 1;
empty tag values remain unknown/not annotated rather than being rewritten as 0.
Lists and nested objects are JSON-encoded inside CSV cells; use `json.loads`.

| Fields | Meaning |
|---|---|
| sample_id | Deterministic identity from the source-relative path, not content |
| filename, image_path, split | Current output name, root-relative path, unchanged split |
| class, hazard_label | Current label: non_hazard=0, hazard=1 |
| group_* | Four independent 0/1 memberships, ordered by label_schema.json |
| tag_groups, num_tag_groups | JSON list of positive group IDs and count |
| label_basis, label_rule_version | Why this label was assigned; rule version |
| original_class, original_filename | Class/name immediately before this build, from sidewalk-data-2 |
| source_dataset, source_path, source_image_path | Absolute source root/file and source-root-relative path |
| source_manifest, source_manifest_csv_line, source_manifest_present | Exact prior CSV record; line includes header; blank/0 for newly discovered files |
| metadata_origin | source_split_csv or discovered_manual_collection |
| is_manual_collection, collection_batch | Ordinary-sidewalk collection indicator and nested batch folder |
| city, city_basis, source_label_id | City/label ID parsed from source filename, or explicitly supplied Pittsburgh collection |
| pre_review_class, pre_review_image_path, applied_manual_class, manual_label_updated_at | Applied historical no_obstacles review, when available |
| positive_tags, positive_hazard_tags | Original positive tags and subset that triggers the new rule |
| annotation_status, num_observed_tag_fields | Distinguish missing annotations, annotated zeros, positive tags and manual nulls |
| normalized_x, normalized_y | Unmodified coordinates in original source-image orientation, not normalized output coordinates |
| 46 original tag columns | Verbatim strings from the source CSV, including blanks |
| source_file_sha256, source_rgb_pixel_sha256 | File hash and hash of `RGB:WIDTHxHEIGHT:` + EXIF-oriented RGB bytes |
| source_bytes, source_mtime_ns | Source size and modification time captured during inventory |
| source_width, source_height, source_mode, source_format, source_exif_orientation, source_alpha_min | Original image properties; alpha_min <255 flags transparency |
| oriented_width, oriented_height | Dimensions after applying EXIF orientation |
| crop_left, crop_top, crop_right, crop_bottom | Full-image rectangle after orientation; right/bottom exclusive |
| preprocess_policy | Versioned preprocessing identifier |
| output_width, output_height, output_mode, output_format, output_bytes | Output image properties |
| output_file_sha256, output_rgb_pixel_sha256 | Encoded output and RGB pixel hashes |
| quality_flags | JSON list of audit flags, not automatic exclusions |
| original_metadata_json | Complete original source split-CSV row; {} for files absent from source CSV |
| source_provenance_json | Related source CSV/review records including source filenames and CSV line numbers |

Manifests contain all images, including non-hazards, and no duplicated rows for
multi-label membership. Count a group as the union of its constituent tag positives.
Do not add tag counts to estimate unique images. Archived manual reviews are provenance,
not training targets under the current tag rule. No image content was auto-annotated.
''')


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source',type=Path,default=Path('/data/datasets/sidewalk-data-2'))
 ap.add_argument('--output',type=Path,default=Path('/data/datasets/sidewalk-data-3'))
 ap.add_argument('--reference-processed',type=Path,default=Path('/data/datasets/proc-sidewalk-data-naive-merged-nulls'))
 ap.add_argument('--workers',type=int,default=16)
 a=ap.parse_args();source=a.source.resolve();target=a.output.resolve();started=now()
 assert source.is_dir() and not target.exists(),'Source must exist; output must not exist.'
 assert not target.is_relative_to(source),'Output must be outside source.'
 assert len(HAZARD_TAGS)==17 and len(TAGS)==46
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 stage=target.parent/('.'+target.name+'.building-'+stamp);stage.mkdir()
 for d in ['audit','statistics','scripts','provenance/source_metadata']:[(stage/d).mkdir(parents=True,exist_ok=True)]
 for split in SPLITS:
  for label in ['hazard','non_hazard']:(stage/split/label).mkdir(parents=True)
 shutil.copy2(__file__,stage/'scripts/build_dataset.py')
 verifier=Path(__file__).with_name('verify_dataset.py')
 if verifier.exists():shutil.copy2(verifier,stage/'scripts/verify_dataset.py')
 print('INVENTORY',source,'STAGING',stage,flush=True)
 records,ignored,coverage=source_inventory(source);provenance,meta=load_provenance(source)
 print('SOURCE_RECORDS',len(records),'COVERAGE',js(coverage),flush=True)
 snapshots=sorted(set(meta+[source/(s+'.csv') for s in SPLITS]+[p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.md','.json')]))
 checksums=[]
 for p in snapshots:
  rel=p.relative_to(source);dest=stage/'provenance/source_metadata'/rel;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest)
  h=filehash(p);assert h==filehash(dest);checksums.append({'source_path':str(p),'snapshot_path':str(dest.relative_to(stage)),'sha256':h,'bytes':p.stat().st_size})
 refcfg=a.reference_processed/'config.yaml'
 if refcfg.exists():shutil.copy2(refcfg,stage/'provenance/previous_processed_config.yaml')
 writecsv(stage/'audit/source_metadata_checksums.csv',checksums)
 writecsv(stage/'audit/source_manifest_coverage.csv',coverage)
 writecsv(stage/'audit/ignored_source_files.csv',ignored,['source_image_path','bytes','reason'])
 writejson(stage/'label_schema.json',{'rule_version':RULE_VERSION,'binary_labels':{'non_hazard':0,'hazard':1},'group_order':list(GROUPS),'group_columns':['group_'+g for g in GROUPS],'groups':{g:{'display_name':DISPLAY[g],'positive_tags':ts} for g,ts in GROUPS.items()},'all_tags':TAGS,'non_hazard_tags':[t for t in TAGS if t not in HAZARD_TAGS],'multi_label':True,'manual_collection_label':'non_hazard','no_positive_hazard_tags_label':'non_hazard','no_tag_annotations_label':'non_hazard'})
 writecsv(stage/'tag_mapping.csv',[{'tag':t,'hazard':int(t in HAZARD_TAGS),'tag_group':next((g for g,ts in GROUPS.items() if t in ts),'')} for t in TAGS])
 rows=[];start=time.monotonic()
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,row in enumerate(pool.map(lambda r:process_one(r,source,stage,provenance),records),1):
   rows.append(row)
   if i%1000==0:print('PROCESSED',i,'/',len(records),'seconds',round(time.monotonic()-start),flush=True)
 for split in SPLITS:writecsv(stage/(split+'.csv'),[r for r in rows if r['split']==split])
 writecsv(stage/'manifest.csv',rows)
 fields=list(rows[0])
 writecsv(stage/'audit/new_images_without_source_manifest.csv',[r for r in rows if not r['source_manifest_present']],fields)
 writecsv(stage/'audit/label_review_candidates.csv',[r for r in rows if 'original_hazard_class_without_tags_assigned_non_hazard' in r['quality_flags']],fields)
 writecsv(stage/'audit/images_without_tag_annotations.csv',[r for r in rows if r['annotation_status']=='no_tag_annotations'],fields)
 print('VALIDATING_ALL_OUTPUTS',flush=True)
 validation=validate(rows,records,source,stage,a.workers)
 for r in checksums:assert filehash(r['source_path'])==r['sha256'],'Source metadata changed during build'
 duplicates=duplicate_audit(rows,stage);stats=summarize(rows,stage)
 print('DRAWING_HISTOGRAMS',flush=True);mpl_version=plots(stats,stage)
 writejson(stage/'audit/duplicate_summary.json',duplicates)
 writejson(stage/'audit/validation.json',validation)
 docs(stage,source,target,rows,stats,validation,duplicates,started)
 writejson(stage/'build_config.json',{'dataset':'sidewalk-data-3','source':str(source),'output':str(target),'built_on_host':platform.node(),'started_at_utc':started,'completed_at_utc':now(),'python':sys.version,'pillow':pillow_version,'matplotlib':mpl_version,'workers':a.workers,'image_count':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),'binary_label_counts':dict(Counter(r['class'] for r in rows)),'manual_collection_count':sum(r['is_manual_collection'] for r in rows),'new_manifest_records':sum(not r['source_manifest_present'] for r in rows),'label_rule_version':RULE_VERSION,'preprocess':{'exif_transpose':True,'mode':'RGB','size':[224,224],'resample':'BICUBIC','crop':'full oriented image','resize_aspect':'stretch','format':'PNG','png_compress_level':6,'model_normalization':False},'split_policy':'preserve_source','duplicate_policy':'preserve_and_audit','source_metadata_hashes':checksums,'build_script_sha256':filehash(__file__),'validation':validation,'duplicate_summary':duplicates})
 if verifier.exists():
  print('INDEPENDENT_VERIFICATION',flush=True)
  subprocess.run([sys.executable,str(stage/'scripts/verify_dataset.py'),str(stage),'--workers',str(a.workers)],check=True)
 writejson(stage/'COMPLETE.json',{'completed_at_utc':now(),'images':len(rows),'validation_passed':True,'manifest_sha256':filehash(stage/'manifest.csv')})
 assert not target.exists();stage.rename(target)
 print('COMPLETE',target,'images',len(rows),'seconds',round(time.monotonic()-start),'COUNTS',js(stats['overlap']),flush=True)

if __name__=='__main__':main()
