from pathlib import Path
from dataclasses import dataclass
import argparse,hashlib,io,json,platform,zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
DEFAULT_ARCHIVE=REPO/'notebooks/colab_results/ensemble/128_bootstrap.zip'
DEFAULT_OUTPUT=HERE/'results'
SPLITS=('pitt-val','pitt-test')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def plain(x):
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=plain,allow_nan=False)+'\n')


def save_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def pct(value):return f'{100*float(value):.2f}%'


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--k',type=int,default=46,help='Fixed review budget, consistent with previous precision@46.')
    return p


@dataclass
class Split:
    name:str
    frame:pd.DataFrame
    p:np.ndarray
    member_ids:np.ndarray

    @property
    def ids(self):return self.frame.sample_id.to_numpy(dtype=str)
    @property
    def y(self):return self.frame.hazard_label.to_numpy(dtype=np.int64)


class Dataset:
    def __init__(self,path):
        self.path=Path(path)
        self.archive_sha256=sha256(self.path)
        with zipfile.ZipFile(self.path) as z:
            roots={n.split('/')[0] for n in z.namelist()}
            assert len(roots)==1,'Archive must contain a single run.'
            self.root=next(iter(roots))
            self.inventory=json.loads(z.read(self.root+'/archive_manifest.json'))
        self.config=json.loads(self.read('config.json'))
        self.status=json.loads(self.read('status.json'))
        self.schema=json.loads(self.read('prediction_schema.json'))
        assert self.config['settings']['n_members']==128
        assert self.status['members_completed']==128 and self.status['status']=='complete'
        assert self.schema['analysis_performed'] is False and self.schema['n_members']==128
        self.checked_files={}

    def read(self,relative):
        with zipfile.ZipFile(self.path) as z:payload=z.read(self.root+'/'+relative)
        expected=self.inventory[relative]
        assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256'],f'Corrupt archive member: {relative}'
        return payload

    def load(self,split):
        assert split in SPLITS,'Held-out ranking analysis uses only pitt-val and pitt-test.'
        frame=pd.read_csv(io.BytesIO(self.read(split+'_predictions.csv')),keep_default_na=False)
        with np.load(io.BytesIO(self.read(split+'_probabilities.npz')),allow_pickle=False) as saved:
            p=saved['probabilities'].astype(np.float64)
            ids=saved['sample_ids'].astype(str)
            members=saved['member_ids'].copy()
        assert frame.sample_id.is_unique and np.array_equal(ids,frame.sample_id.astype(str))
        assert np.array_equal(members,np.arange(1,129)) and p.shape==(len(frame),128)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        columns=[f'p_hazard_head_{j:03d}' for j in range(1,129)]
        assert np.allclose(frame[columns].to_numpy(float),p,atol=5e-8,rtol=1e-7)
        manifest_bytes=self.read('dataset_metadata/'+split+'.csv')
        assert hashlib.sha256(manifest_bytes).hexdigest()==self.config['expected_dataset_hashes'][split+'.csv']
        manifest=pd.read_csv(io.BytesIO(manifest_bytes),keep_default_na=False)
        assert np.array_equal(manifest.sample_id.astype(str),ids)
        for column in ['image_path','hazard_label','output_rgb_pixel_sha256']:
            assert np.array_equal(frame[column],manifest[column])
        assert set(frame.split)=={split} and set(frame.hazard_label)=={0,1}
        self.checked_files[split]=dict(n_images=len(frame),hazards=int(frame.hazard_label.sum()),
                                      manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        return Split(split,frame,p,members)

    def provenance(self):
        import scipy
        return dict(archive=str(self.path),archive_sha256=self.archive_sha256,run_id=self.root,
                    verified_splits=self.checked_files,n_members=128,configuration=self.config,
                    input_verification='Archive-member SHA-256; NPZ/CSV and dataset-manifest identity/label checks.',
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                    script_sha256={p.name:sha256(p) for p in HERE.glob('*.py')},
                    reference_pdf_sha256=sha256(HERE/'reference/uncertainty_in_classifier_rankings.pdf'))


def probability_summary(data):
    frame=data.frame[[c for c in ['sample_id','image_path','split','hazard_label','original_class','city','is_manual_collection','output_rgb_pixel_sha256'] if c in data.frame]].copy()
    frame['p_hazard_mean']=data.p.mean(1)
    frame['p_hazard_sd']=data.p.std(1,ddof=1)
    frame['p_hazard_q025']=np.quantile(data.p,.025,axis=1,method='linear')
    frame['p_hazard_q975']=np.quantile(data.p,.975,axis=1,method='linear')
    return frame


def order_scores(scores,ids,descending=True):
    score=np.asarray(scores,dtype=float)
    assert score.ndim==1 and np.isfinite(score).all()
    return np.lexsort((np.asarray(ids,dtype=str),-score if descending else score))


def rank_metrics(y,order,k):
    y=np.asarray(y,dtype=int);order=np.asarray(order,dtype=int)
    assert 0<k<=len(y) and set(np.unique(y))=={0,1}
    assert np.array_equal(np.sort(order),np.arange(len(y)))
    hits=int(y[order[:k]].sum())
    return dict(n_images=len(y),hazards=int(y.sum()),k=int(k),selected_fraction=k/len(y),
                hits=hits,precision_at_k=hits/k,recall_at_k=hits/int(y.sum()))


def export_ranking(folder,data,name,order,extra=None,k=46):
    frame=probability_summary(data)
    if extra:
        for column,values in extra.items():frame[column]=values
    frame=frame.iloc[order].reset_index(drop=True);frame.insert(0,'rank',np.arange(1,len(frame)+1))
    save_csv(Path(folder)/(data.name+'_'+name+'.csv'),frame)
    save_csv(Path(folder)/(data.name+'_'+name+f'_top{k}.csv'),frame.iloc[:k])
    return rank_metrics(data.y,order,k)


def pairwise_counts(p):
    p=np.asarray(p,dtype=float);n,b=p.shape
    assert np.isfinite(p).all()
    wins=np.zeros((n,n),dtype=np.int32)
    for member in range(b):wins+=(p[:,member,None]>p[None,:,member])
    ties=b-wins-wins.T
    assert (ties>=0).all() and np.all(np.diag(wins)==0) and np.all(np.diag(ties)==b)
    return wins,ties


def pairwise_exports(folder,data,wins,ties):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    p=wins/data.p.shape[1];t=ties/data.p.shape[1]
    np.savez_compressed(folder/(data.name+'_pairwise.npz'),p_strict=p,p_tie=t,wins=wins,
                        ties=ties,sample_ids=data.ids,member_ids=data.member_ids)
    pd.DataFrame(p,index=pd.Index(data.ids,name='row_sample_id'),columns=data.ids).to_csv(folder/(data.name+'_pairwise_probability.csv'))
    pd.DataFrame(t,index=pd.Index(data.ids,name='row_sample_id'),columns=data.ids).to_csv(folder/(data.name+'_pairwise_tie_probability.csv'))
    return dict(unordered_pairs=len(p)*(len(p)-1)//2,
                pairs_with_any_score_tie=int(np.triu(ties>0,1).sum()),
                pairs_tied_in_all_heads=int(np.triu(ties==data.p.shape[1],1).sum()),
                equal_pairwise_preference_pairs=int(np.triu(wins==wins.T,1).sum()),
                uncertain_preference_pairs=int(np.triu((p+.5*t>=.4)&(p+.5*t<=.6),1).sum()))


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig,path):
    path=Path(path)
    for suffix in ['.png','.pdf']:fig.savefig(path.with_suffix(suffix),dpi=180,bbox_inches='tight')


def results_table(rows,k=46):
    return md_table(['Split','Ranking',f'Hazards in top {k}',f'Precision@{k}',f'Recall@{k}'],
                    [[split,name,f"{m['hits']}/{m['k']}",pct(m['precision_at_k']),pct(m['recall_at_k'])] for split,name,m in rows])


def shared_notes(data,k):
    return (f"Source: `{data.path}`; run `{data.root}`; archive SHA-256 `{data.archive_sha256}`. "
            "The source model is frozen DINOv3 ViT-B/16 with 128 independently fitted linear heads. "
            "Each head uses a replacement bootstrap of the 1,276 Pitt training records (527 hazards, 749 non-hazards), "
            "sampling exact-pixel clusters stratified by binary-label composition. Other-city training images are not used. "
            "Settings are Adam, learning rate 3e-5, weight decay 0.01, batch size 64, "
            "5,000 updates, unweighted binary cross-entropy, final checkpoint. "
            f"We keep the historical fixed budget k={k}: "
            f"{100*k/382:.4f}% of 382 validation observations and {100*k/380:.4f}% of 380 test observations. "
            "Test labels are used only to score completed rankings, never to fit lambda or choose the pairwise solver.\n\n"
            "These frequencies and intervals describe variability across fitted bootstrap heads conditional on the "
            "shared frozen representation and dataset. They are not automatically calibrated Bayesian probabilities "
            "or 95% coverage guarantees. Exact duplicate images/annotation conflicts are retained from the dataset. "
            "Training predictions are excluded from these held-out comparisons because they include in-bag fits.\n")
