"""Offline inference for a saved sidewalk-data-3 DINOv3 inference/ folder.
Example: python sidewalk_inference.py --bundle ./inference --images a.jpg b.png --output predictions.csv
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageOps


def load_heads(bundle, device):
    bundle=Path(bundle); heads={}; metadata={}
    for task in ['hazard','tag_groups']:
        saved=torch.load(bundle/('best_'+task+'.pt'),map_location='cpu',weights_only=True)
        assert saved['task']==task
        head=nn.Linear(saved['feature_dim'],len(saved['output_labels'])).to(device)
        head.load_state_dict(saved['state_dict']);head.eval().requires_grad_(False)
        heads[task]=head;metadata[task]=saved
    assert metadata['hazard']['binding']==metadata['tag_groups']['binding']
    return heads,metadata


class SidewalkPredictor:
    def __init__(self,bundle,device=None):
        from transformers import AutoModel
        self.bundle=Path(bundle);self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.settings=json.loads((self.bundle/'preprocessing.json').read_text())
        self.heads,self.metadata=load_heads(self.bundle,self.device)
        self.encoder=AutoModel.from_pretrained(self.bundle/'backbone',local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.mean=torch.tensor(self.settings['mean'],device=self.device).view(1,3,1,1)
        self.std=torch.tensor(self.settings['std'],device=self.device).view(1,3,1,1)
        self.groups=self.metadata['tag_groups']['output_labels']
        self.hazard_threshold=float(self.metadata['hazard']['thresholds'])
        self.group_thresholds=np.asarray(self.metadata['tag_groups']['thresholds'])
        if str(self.device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False

    @torch.inference_mode()
    def predict(self,paths,batch_size=64):
        paths=[Path(p) for p in paths];rows=[]
        for begin in range(0,len(paths),batch_size):
            chosen=paths[begin:begin+batch_size];pixels=[]
            for path in chosen:
                with Image.open(path) as image:
                    image=ImageOps.exif_transpose(image).convert('RGB').resize(tuple(self.settings['size']),Image.Resampling.BICUBIC)
                    pixels.append(np.array(image,copy=True))
            x=torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).to(self.device).float()/255
            x=(x-self.mean)/self.std
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(self.device).startswith('cuda')):
                features=self.encoder(pixel_values=x).last_hidden_state[:,0].float()
            hp=self.heads['hazard'](features).sigmoid().cpu().numpy()[:,0]
            gp=self.heads['tag_groups'](features).sigmoid().cpu().numpy()
            for i,path in enumerate(chosen):
                hazard=bool(hp[i]>=self.hazard_threshold)
                ungated=[g for j,g in enumerate(self.groups) if gp[i,j]>=self.group_thresholds[j]]
                row=dict(image_path=str(path),p_hazard=float(hp[i]),pred_hazard=int(hazard),
                         predicted_class='hazard' if hazard else 'non_hazard',
                         predicted_tag_groups=json.dumps(ungated if hazard else []),
                         ungated_tag_groups=json.dumps(ungated),hazard_without_group=int(hazard and not ungated))
                row.update({'p_'+g:float(gp[i,j]) for j,g in enumerate(self.groups)})
                rows.append(row)
        return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--images',nargs='+',required=True);p.add_argument('--output',type=Path,default=Path('predictions.csv'))
    p.add_argument('--device',default=None);a=p.parse_args()
    model=SidewalkPredictor(a.bundle,a.device);rows=model.predict(a.images)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved',len(rows),'predictions to',a.output)
if __name__=='__main__':main()
