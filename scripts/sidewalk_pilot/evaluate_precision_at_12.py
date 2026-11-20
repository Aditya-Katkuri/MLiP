#!/usr/bin/env python3
"""Score blinded review labels after labeling, without treating model predictions as truth.
Usage: python evaluate_precision_at_12.py --labels reviewer.csv --root /data/datasets/sidewalk-pilot --output evaluation.csv
"""
import argparse,csv,json
from pathlib import Path

def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def parse_label(raw):
 s=str(raw).strip().lower()
 if s in {'1','hazard','hazardous','yes'}:return 1
 if s in {'0','non_hazard','non-hazard','nonhazard','no'}:return 0
 if s in {'','uncertain','unknown','unsure'}:return None
 raise ValueError(f'Unexpected hazard label: {raw!r}')

def score(top, labels, k):
 assert len(top)==k
 known=[labels.get((r['stretch'],int(r['image_number']))) for r in top]
 positives=sum(x==1 for x in known);unknown=sum(x is None for x in known)
 return {'k':k,'top_k_labeled':k-unknown,'top_k_hazards':positives,'top_k_unknown':unknown,
         'precision_at_12':positives/k if not unknown else '',
         'precision_at_12_lower_bound':positives/k,'precision_at_12_upper_bound':(positives+unknown)/k}

def evaluate(root, review_rows):
 mapping=read(Path(root)/'results/review_mapping.csv')
 allowed={(r['stretch'],int(r['image_number'])) for r in mapping};labels={}
 for row in review_rows:
  key=(row['stretch'].strip(),int(row['image_number']))
  if key not in allowed:raise ValueError(f'Unknown image: {key}')
  if key in labels:raise ValueError(f'Duplicate review label: {key}')
  labels[key]=parse_label(row['hazard_label'])
 output=[]
 for stretch in 'ABCD':
  rs=[r for r in mapping if r['stretch']==stretch]
  presented=sorted(rs,key=lambda r:int(r['image_number']))
  natural=sorted(rs,key=lambda r:int(r['natural_order']))
  variants=[('presented_order',presented)]
  if stretch in 'BD':variants.append(('natural_order_same_images',natural))
  all_labels=[labels.get((r['stretch'],int(r['image_number']))) for r in rs]
  labeled=sum(x is not None for x in all_labels);hazards=sum(x==1 for x in all_labels)
  for ordering,ordered in variants:
   output.append({'stretch':stretch,'ordering':ordering,'images':len(rs),'images_labeled':labeled,
     'hazards_labeled':hazards,'all_images_hazard_prevalence':hazards/len(rs) if labeled==len(rs) else '',
     **score(ordered[:12],labels,12)})
 return output

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--labels',type=Path,required=True)
 ap.add_argument('--root',type=Path,default=Path('/data/datasets/sidewalk-pilot'))
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=evaluate(a.root,read(a.labels));a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
