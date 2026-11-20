from pathlib import Path
import json,hashlib,textwrap
P=Path(__file__).parent
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
def code(s,hidden=False):cells.append(dict(cell_type='code',metadata={'jupyter':{'source_hidden':True}} if hidden else {},execution_count=None,outputs=[],source=textwrap.dedent(s).strip()+'\n'))
core=(P/'sidewalk_dino_core.py').read_text();inference=(P/'sidewalk_inference.py').read_text()
hashes=json.loads((P/'dataset_hashes.json').read_text())
md('''
# DINOv3 on sidewalk-data-3: hazard detection and four tag groups

Use a **Colab GPU runtime (T4 is sufficient)**. Run the cells in order and mount Google Drive.
This notebook is separate from `dino_ft_A_report_rerun.ipynb` and uses the new dataset's saved labels.

**Before Run All:** copy the server archive `/data/datasets/sidewalk-data-3.tar` to
`MyDrive/MLiP/datasets/sidewalk-data-3.tar` (or configure the Hugging Face option below).
The archive is about 2.47 GB. Supply an `HF_TOKEN` with access to
`facebook/dinov3-vitb16-pretrain-lvd1689m` through Colab Secrets, the environment, or the hidden prompt.
No token is embedded or saved in this notebook.

**Experiment:** frozen DINOv3 ViT-B/16, CLS features, a linear layer with five independent logits:
one hazard score and four tag-group scores. The hazard loss uses all images; group loss uses only
true hazards. Images can have multiple groups. The classes/tags are read from the dataset, never
inferred from the old Obstacle/SurfaceProblem names.

**Selection:**
1. Across the three seeds, maximize mean validation **precision@46**; break ties by the best mean
   validation **hazard balanced accuracy**, choosing the hazard threshold from `[0.3, 0.5, 0.7]`.
2. Independently choose the configuration with best mean validation **macro tag-group balanced
   accuracy**, on **all true hazards**, including those missed by stage 1. For each of the four
   groups, BA = (sensitivity + specificity)/2; macro BA is their unweighted mean. Choose each
   group's threshold from the same grid using validation means over seeds.
3. Save all three replicas of each winning configuration. For a single deployable model per task,
   choose the best validation replica within that configuration using the same objective; ties
   use the smaller seed. Report this deployment model separately from the three-seed mean.

The two training orders, seven learning rates, two weight decays and three seeds give **84 trials**,
not 168. Each trial trains all five linear outputs. Compare final checkpoints after 5,000 Pitt updates,
or 5,000 Other + 5,000 Pitt updates. Threshold searches do not add training runs. Test performance
is computed only after selection is frozen; it never affects the winning configurations, seeds or thresholds.

All artifacts are saved to `MyDrive/MLiP/dino_sidewalk_data_3_runs/<run-id>/`.
Resume by setting `RESUME_RUN_DIR` to that directory. Completed trials and feature caches are reused;
an interrupted trial restarts. Resume requires the same code, settings and feature-extraction environment.
''')
code('''
%pip install -q "transformers==5.22.1" pandas matplotlib huggingface_hub pillow
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch, transformers
assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > T4 GPU, then rerun.'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__, '| transformers:', transformers.__version__)
''')
md('''## Configuration
The default uses the prepared archive in Google Drive. The alternative Hugging Face mode requires
the repository and archive filename for **dataset 3**, not the old merged-nulls dataset.
Do not change hyperparameters while resuming; use a fresh run directory instead.''')
code('''
from pathlib import Path
DATA_SOURCE = 'drive'  # 'drive', 'huggingface', or 'existing' (already extracted under DATA_ROOT)
DRIVE_ARCHIVE = Path('/content/drive/MyDrive/MLiP/datasets/sidewalk-data-3.tar')
HF_DATA_REPO = ''       # Required only for DATA_SOURCE='huggingface'
HF_DATA_ARCHIVE = 'sidewalk-data-3.tar'
HF_DATA_REVISION = None # Resolved to an immutable commit on first download
DATA_ROOT = Path('/content/sidewalk-data-3')
DRIVE_ROOT = Path('/content/drive/MyDrive/MLiP/dino_sidewalk_data_3_runs')
RESUME_RUN_DIR = None
RUN_TEST = True
CACHE_BATCH_SIZE = 64

BACKBONE = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
BACKBONE_REVISION = '5931719e67bbdb9737e363e781fb0c67687896bc'
EXPECTED_ARCHIVE_SHA256 = 'd6c88a2b770baefd9ee34b7c846fe8be573258ba6d6ec7cac27e90e2d9eabbbd'
'''+ 'EXPECTED_DATASET_HASHES = '+repr(hashes)+'\n'+'''
SETTINGS = dict(
    train_plans={'pitt': [('pitt-train', 5000)],
                 'other_then_pitt': [('other-cities-train', 5000), ('pitt-train', 5000)]},
    learning_rates=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    weight_decays=[0.0, 0.01], seeds=[0, 1, 2], batch_size=64, eval_every=250,
    thresholds=[0.3, 0.5, 0.7], k=46,
)
''')
md('''## Implementation helpers
This collapsed cell contains the self-contained training, selection, and inference implementations.
They are also saved with the results. No companion file needs to be downloaded into Colab.''')
code("CORE_SOURCE = "+repr(core)+"\nINFERENCE_SOURCE = "+repr(inference)+'''
import importlib.util, sys
MODULE_PATH = Path('/content/sidewalk_dino_core.py')
MODULE_PATH.write_text(CORE_SOURCE)
spec = importlib.util.spec_from_file_location('sidewalk_dino_core', MODULE_PATH)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
''',hidden=True)
md('''## Mount Drive and create a persistent run
This must succeed before feature extraction or training. Configuration identity and saved artifacts
are checked when resuming. Tokens are read lazily and are excluded from saved configuration.''')
code('''
import json, hashlib, shutil, time, tempfile, tarfile, getpass, importlib.metadata
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')
assert Path('/content/drive/MyDrive').is_dir()

HF_TOKEN = None
def get_hf_token():
    global HF_TOKEN
    if HF_TOKEN: return HF_TOKEN
    HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()
    if not HF_TOKEN:
        try:
            from google.colab import userdata
            HF_TOKEN = (userdata.get('HF_TOKEN') or '').strip()
        except Exception:
            HF_TOKEN = ''
    if not HF_TOKEN: HF_TOKEN = getpass.getpass('HF read token with DINOv3 access: ').strip()
    if not HF_TOKEN: raise RuntimeError('A token with access to the gated DINOv3 backbone is required.')
    return HF_TOKEN

run_config = dict(version='dino-sidewalk-data-3-v1', settings=SETTINGS,
    backbone=BACKBONE, backbone_revision=BACKBONE_REVISION, pooling='cls', backbone_frozen=True,
    expected_dataset_hashes=EXPECTED_DATASET_HASHES, group_order=core.GROUPS,
    core_code_sha256=hashlib.sha256(CORE_SOURCE.encode()).hexdigest(),
    inference_code_sha256=hashlib.sha256(INFERENCE_SOURCE.encode()).hexdigest(),
    feature_precision='float16', head_precision='float32',
    preprocessing=dict(size=[224,224],resize='full_image_bicubic',mode='RGB',mean=core.MEAN,std=core.STD),
    checkpoint_selection='final_step_only', optimizer='Adam',
    loss='unweighted hazard BCE + mean four-group BCE masked to true hazards')
RUN_DIR = Path(RESUME_RUN_DIR) if RESUME_RUN_DIR else DRIVE_ROOT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6])
if RESUME_RUN_DIR:
    assert (RUN_DIR/'config.json').is_file(), 'Resume folder has no config.json.'
    assert core.canonical(json.loads((RUN_DIR/'config.json').read_text())) == core.canonical(run_config), 'Code/settings/data changed; start a new run.'
else:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    core.save_json(RUN_DIR/'config.json',run_config)
probe=RUN_DIR/'.write_probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
(RUN_DIR/'sidewalk_dino_core.py').write_text(CORE_SOURCE)
(RUN_DIR/'sidewalk_inference.py').write_text(INFERENCE_SOURCE)
environment=dict(python=sys.version,gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),
    transformers=transformers.__version__,cuda=str(torch.version.cuda),
    packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','huggingface-hub','Pillow']})
feature_binding=dict(run_config_sha256=core.fingerprint(run_config),backbone_revision=BACKBONE_REVISION,
    gpu=environment['gpu'],torch=environment['torch'],transformers=environment['transformers'],precision='float16',
    torch_tf32=False,cache_batch_size=CACHE_BATCH_SIZE)
if (RUN_DIR/'feature_binding.json').exists():
    assert json.loads((RUN_DIR/'feature_binding.json').read_text())==feature_binding, 'Feature-extraction environment changed; start a new run.'
else:core.save_json(RUN_DIR/'feature_binding.json',feature_binding)
core.save_json(RUN_DIR/'environment.json',environment)
assert len(core.configs(SETTINGS))==84
print('Persistent run:',RUN_DIR)
print('84 final-checkpoint trials; selection uses validation only.')
''')
md('''## Load and verify dataset 3
Extraction and feature reads use Colab's local disk. The server archive and manifests are pinned by
SHA-256. Each image is checked against its recorded hash when features are first computed. All
277 additional Pittsburgh training screenshots are included as non-hazards. Existing annotation
conflicts are retained under dataset 3's documented policy; no silent filtering or deduplication occurs.''')
code('''
from huggingface_hub import HfApi, hf_hub_download
already_extracted = all((DATA_ROOT/name).is_file() for name in EXPECTED_DATASET_HASHES)
if not already_extracted:
    if DATA_ROOT.exists(): raise RuntimeError(f'{DATA_ROOT} is incomplete; choose a fresh DATA_ROOT or remove your incomplete extraction.')
    if DATA_SOURCE == 'drive':
        if not DRIVE_ARCHIVE.is_file():
            raise FileNotFoundError(f'Upload /data/datasets/sidewalk-data-3.tar from the server to {DRIVE_ARCHIVE}, then rerun this cell.')
        local_tar=Path('/content/sidewalk-data-3.tar')
        if not local_tar.exists() or core.file_sha(local_tar)!=EXPECTED_ARCHIVE_SHA256:
            shutil.copyfile(DRIVE_ARCHIVE, local_tar)
        download_info=dict(source='google_drive',archive=str(DRIVE_ARCHIVE))
    elif DATA_SOURCE == 'huggingface':
        assert HF_DATA_REPO, 'Set HF_DATA_REPO to the repository containing sidewalk-data-3.tar.'
        revision=HF_DATA_REVISION or HfApi(token=get_hf_token()).dataset_info(HF_DATA_REPO).sha
        local_tar=Path(hf_hub_download(HF_DATA_REPO,HF_DATA_ARCHIVE,repo_type='dataset',revision=revision,token=get_hf_token()))
        download_info=dict(source='huggingface',repo=HF_DATA_REPO,revision=revision,archive=HF_DATA_ARCHIVE)
    else:
        raise FileNotFoundError('DATA_SOURCE=existing requires the fully extracted dataset at DATA_ROOT.')
    assert core.file_sha(local_tar)==EXPECTED_ARCHIVE_SHA256, 'Archive is not the prepared dataset-3 snapshot.'
    DATA_ROOT.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DATA_ROOT.parent,prefix='sidewalk3-extract-') as temp:
        with tarfile.open(local_tar) as archive:
            for member in archive.getmembers():
                path=Path(member.name)
                assert not path.is_absolute() and '..' not in path.parts and path.parts[0]=='sidewalk-data-3'
                assert member.isfile() or member.isdir(), 'Archive contains a non-regular entry.'
            archive.extractall(temp,filter='data')
        Path(temp,'sidewalk-data-3').rename(DATA_ROOT)
    core.save_json(RUN_DIR/'download.json',{**download_info,'sha256':EXPECTED_ARCHIVE_SHA256})

data,schema=core.load_data(DATA_ROOT,EXPECTED_DATASET_HASHES)
assert {s:len(df) for s,df in data.items()}=={'other-cities-train':22912,'pitt-train':1276,'pitt-val':382,'pitt-test':380}
snapshot=RUN_DIR/'dataset_metadata';snapshot.mkdir(exist_ok=True)
for name in list(EXPECTED_DATASET_HASHES)+['README.md','LABELING_RULES.md','PREPROCESSING.md','build_config.json','tag_mapping.csv']:
    shutil.copy2(DATA_ROOT/name,snapshot/name)
for folder in ['statistics','audit']:
    for name in (['binary_class_counts.csv','tag_group_counts.csv','tag_counts_by_split.csv'] if folder=='statistics' else ['duplicate_label_summary.json','release_validation.json','label_review_candidates.csv']):
        shutil.copy2(DATA_ROOT/folder/name,snapshot/name)
summary=pd.DataFrame([dict(split=s,images=len(df),hazard=int(df.hazard_label.sum()),non_hazard=int((df.hazard_label==0).sum()),manual_nulls=int(df.is_manual_collection.sum())) for s,df in data.items()])
core.save_csv(RUN_DIR/'dataset_summary.csv',summary)
display(summary)
print('Group output order:',core.GROUPS)
''')
md('''## Frozen backbone and cached training/validation features
The encoder uses the exact pinned DINOv3 revision. Features use FP16 on GPU; linear heads train
in FP32. Images are already RGB 224×224, so only ImageNet normalization is applied here—no extra
crop or resize. The frozen backbone is saved once to Drive for later offline inference.
Test features are deliberately deferred until model selection has finished.''')
code('''
from transformers import AutoModel
encoder=AutoModel.from_pretrained(BACKBONE,revision=BACKBONE_REVISION,token=get_hf_token()).to('cuda').eval().requires_grad_(False)
assert not any(p.requires_grad for p in encoder.parameters())
features={}
for split in ['other-cities-train','pitt-train','pitt-val']:
    features[split]=core.cache_features(encoder,data[split],split,DATA_ROOT,RUN_DIR,feature_binding,CACHE_BATCH_SIZE,'cuda')

inference_dir=RUN_DIR/'inference';inference_dir.mkdir(exist_ok=True)
backbone_dir=inference_dir/'backbone'
if backbone_dir.exists():
    assert (inference_dir/'backbone_files.json').exists(), 'Incomplete backbone export.'
    for relative,expected in json.loads((inference_dir/'backbone_files.json').read_text()).items():
        assert core.file_sha(backbone_dir/relative)==expected
else:
    pending=inference_dir/'.backbone_pending'
    if pending.exists(): shutil.rmtree(pending)  # only this run's interrupted export
    with tempfile.TemporaryDirectory() as temp:
        encoder.save_pretrained(temp,safe_serialization=True)
        shutil.copytree(temp,pending)
    hashes={str(p.relative_to(pending)):core.file_sha(p) for p in pending.rglob('*') if p.is_file()}
    core.save_json(inference_dir/'backbone_files.json',hashes)
    pending.rename(backbone_dir)
print('Backbone and feature caches saved on Drive.')
''')
md('''## Run the 84 trials
The two training orders share the same settings. Other→Pitt continues the same head and Adam
optimizer; it does not reset either at the phase boundary. Loss curves are recorded every 250
updates. These intermediate metrics use 0.5 for display only; selection searches the configured
threshold grid after training. Only final-update checkpoints compete for selection.''')
code('''
try:
    entries,history=core.run_sweep(SETTINGS,feature_binding,features,data,RUN_DIR)
except BaseException as error:
    core.save_json(RUN_DIR/'status.json',dict(status='interrupted',error_type=type(error).__name__,message=str(error)))
    raise
print('All trials are saved:',RUN_DIR/'trials')
''')
md('''## Select the two configurations on validation
Precision ties are determined using integer summed top-46 hits across seeds, avoiding accidental
floating-point tie-breaking. Balanced accuracy breaks these ties after threshold selection.
Remaining exact ties prefer Pitt-only, then lower learning rate, then lower weight decay.
Exact threshold ties choose the lower threshold. Group selection is independent of hazard selection.

Four threshold searches are sufficient for the group task: macro balanced accuracy is the sum of
four independent per-group scores. This is equivalent to checking all 3⁴ threshold combinations
for the selected objective, without changing training or using test data.''')
code('''
selection,validation_summary,hazard_thresholds,group_thresholds=core.select_models(
    entries,data['pitt-val'],SETTINGS['thresholds'],SETTINGS['seeds'],SETTINGS['k'])
core.save_json(RUN_DIR/'selection.json',selection)
core.save_csv(RUN_DIR/'validation_configurations.csv',validation_summary)
core.save_csv(RUN_DIR/'validation_hazard_thresholds.csv',hazard_thresholds)
core.save_csv(RUN_DIR/'validation_group_thresholds.csv',group_thresholds)
frames=[]
for entry in entries:
    frame=data['pitt-val'][['sample_id','image_path','hazard_label','original_class']+core.GROUP_COLUMNS].copy()
    frame['trial_id']=entry['trial_id'];frame['config_id']=core.config_id(entry['cfg']);frame['seed']=entry['cfg']['seed']
    frame['p_hazard']=entry['val_probabilities'][:,0]
    for j,g in enumerate(core.GROUPS):frame['p_'+g]=entry['val_probabilities'][:,j+1]
    frames.append(frame)
core.save_csv(RUN_DIR/'validation_all_predictions.csv',pd.concat(frames,ignore_index=True))
print('All 28 configurations at their selected validation thresholds:')
display(validation_summary.sort_values(['precision_at_k_mean','hazard_ba_mean'],ascending=False))
print('Chosen hazard configuration:',selection['hazard'])
print('Chosen tag-group configuration:',selection['tag_groups'])
''')
md('''## Save deployable weights and verify loading
`inference/best_hazard.pt` and `inference/best_tag_groups.pt` contain the independently selected
heads, thresholds and label order. `inference/backbone/` contains the complete frozen encoder.
All three winning-configuration replicas are retained under `inference/replicas/`.
The exported heads are reloaded and checked against the original validation probabilities before
test evaluation. The chosen single replicas and the three-seed configuration means are different
quantities and are reported separately.''')
code('''
inference_dir=core.export_selection(selection,entries,schema,feature_binding,RUN_DIR,features['pitt-val'])
(inference_dir/'sidewalk_inference.py').write_text(INFERENCE_SOURCE)
core.save_json(inference_dir/'preprocessing.json',dict(size=[224,224],resize='full_image_bicubic',exif_transpose=True,
    mode='RGB',scale=255,mean=core.MEAN,std=core.STD,pooling='cls',gpu_precision='float16',head_precision='float32',
    backbone=BACKBONE,backbone_revision=BACKBONE_REVISION))
requirements='\\n'.join(['torch=='+str(torch.__version__).split('+')[0], 'transformers=='+transformers.__version__,
    'numpy=='+importlib.metadata.version('numpy'),'Pillow=='+importlib.metadata.version('Pillow')])+'\\n'
(inference_dir/'requirements.txt').write_text(requirements)
(inference_dir/'README.md').write_text('''+repr('''# Saved sidewalk-data-3 models

Keep this entire inference/ directory. No Hugging Face token or network download is needed
when loading the complete bundle: backbone/ contains the frozen encoder weights.

Run `python sidewalk_inference.py --bundle . --images image1.png image2.jpg --output predictions.csv`.
Install requirements.txt using a PyTorch build appropriate for the inference device.

best_hazard.pt and best_tag_groups.pt are the independently selected single deployment
replicas from the validation-winning configurations. selection.json records all hyperparameters,
seeds and thresholds. replicas/ retains all three seeds of both selected configurations.
Reported three-seed means are not ensemble metrics or the single-model deployment scores.
No test score was used to choose these files. checkpoint_reload_verification.json verifies the
exported head probabilities on the original validation features.

Input: any ordinary image; EXIF orientation, RGB conversion, full-frame bicubic resize to
224x224, and saved ImageNet normalization are applied identically to the dataset pipeline.
For already processed dataset images the resize is an identity operation.

Outputs: p_hazard, binary decision, four conditional tag-group probabilities and multiple
group decisions. Group probabilities need not sum to one. Pipeline group decisions are
suppressed for a predicted non-hazard. If a hazard has no group above its threshold,
hazard_without_group=1; no arbitrary argmax or forced group is introduced.
Group evaluation in the experiment uses ALL true hazards, regardless of stage 1 predictions.
The separate gated pipeline metrics use all images and include stage 1 errors.

CPU inference uses float32 backbone operations, while the T4 experiment uses float16 features;
small numerical differences across hardware/precision are possible. Model weights and the
complete preprocessing specification are saved for reproducibility.
''')+''')
core.save_json(RUN_DIR/'status.json',dict(status='selected_models_exported',trials_completed=len(entries)))
print('Inference weights:',inference_dir)
print((inference_dir/'checkpoint_reload_verification.json').read_text())
''')
md('''## Validation summaries and learning curves
The configuration table above uses optimized validation thresholds. The subtype learning curve
below uses 0.5 only as a consistent training-time diagnostic; it is not the selection score.''')
code('''
validation_metrics,validation_deployment=core.evaluate_selected(selection,entries,features,data,RUN_DIR,'pitt-val')
display(validation_metrics)
print('Single deployment-model validation scores:')
print('Hazard:',validation_deployment['hazard'])
print('Groups:',validation_deployment['groups_on_all_true_hazards'])
core.plot_curves(history,selection,RUN_DIR)
''')
md('''## Test the frozen selections
No threshold, configuration, seed, or checkpoint is selected here. Save per-image probabilities,
per-group precision/recall/F1/balanced accuracy, confusion counts, the three-seed summaries, and
separate single-deployment-model and gated-pipeline metrics. The test labels are the new dataset-3
labels, so the old report's test scores and old two-subtype definition do not apply.''')
code('''
selection_hash=core.file_sha(RUN_DIR/'selection.json')
if RUN_TEST:
    selection=json.loads((RUN_DIR/'selection.json').read_text())
    features['pitt-test']=core.cache_features(encoder,data['pitt-test'],'pitt-test',DATA_ROOT,RUN_DIR,feature_binding,CACHE_BATCH_SIZE,'cuda')
    test_metrics,test_deployment=core.evaluate_selected(selection,entries,features,data,RUN_DIR,'pitt-test')
    assert core.file_sha(RUN_DIR/'selection.json')==selection_hash
    display(test_metrics)
    print('Single deployment-model test hazard scores:',test_deployment['hazard'])
    print('Single deployment-model test tag-group scores:',test_deployment['groups_on_all_true_hazards'])
else:
    print('Test disabled. The selected inference weights are already saved.')
''')
md('''## Check offline image inference and finish
Reload the exported bundle and predict a few images without contacting Hugging Face. This cell
checks the inference entry point; it does not change selection. Keep the entire Drive run folder,
especially `inference/`, for later inference and reporting.''')
code('''
spec=importlib.util.spec_from_file_location('sidewalk_inference',inference_dir/'sidewalk_inference.py')
infer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(infer_module)
# Free the training encoder before loading its saved offline copy.
del encoder
torch.cuda.empty_cache()
predictor=infer_module.SidewalkPredictor(inference_dir,device='cuda')
example_split='pitt-test' if RUN_TEST else 'pitt-val'
examples=[DATA_ROOT/p for p in data[example_split].image_path.iloc[:3]]
preview=pd.DataFrame(predictor.predict(examples))
core.save_csv(RUN_DIR/'offline_inference_example.csv',preview)
display(preview)
core.save_json(RUN_DIR/'status.json',dict(status='complete' if RUN_TEST else 'complete_without_test',
    trials_completed=len(entries),inference_bundle=str(inference_dir),offline_inference_checked=True,test_used_for_selection=False))
# A compact checksum manifest covers all saved inference files, including encoder weights.
core.save_json(inference_dir/'files_sha256.json',{str(p.relative_to(inference_dir)):core.file_sha(p)
    for p in inference_dir.rglob('*') if p.is_file() and p.name!='files_sha256.json'})
print('Finished. Complete run:',RUN_DIR)
print('Deployable models:',inference_dir)
''')
md('''## Saved outputs

- `config.json`, `environment.json`, `feature_binding.json`, `dataset_metadata/`: settings and data identity.
- `trials/`: all 84 final weights, validation probabilities, and learning histories; resumable.
- `validation_configurations.csv`: all 28 configurations with their selected thresholds and seed means/std.
- `validation_hazard_thresholds.csv`, `validation_group_thresholds.csv`: every threshold and seed score.
- `selection.json`: exact selected configurations, thresholds, replicas, and selection rules.
- `pitt-val_*`, `pitt-test_*`: per-image predictions, per-group metrics, seed summaries and deployment metrics.
- `selected_learning_curves.png/.pdf`: loss and validation metric curves for the winning configurations.
- **`inference/`**: complete backbone, **`best_hazard.pt`**, **`best_tag_groups.pt`**, all selected seed weights,
  thresholds, label schema, preprocessing, checksums, and a standalone offline inference script.

For the group task, “macro balanced accuracy” averages sensitivity and specificity within each
of the four binary group decisions, then averages equally across groups. It is not four-way
multiclass accuracy or macro positive recall. The evaluated population is all true hazards.
[DINOv3 API documentation](https://huggingface.co/docs/transformers/model_doc/dinov3) ·
[Balanced accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11'),colab=dict(provenance=[],gpuType='T4'),accelerator='GPU'),cells=cells)
for i,c in enumerate(cells):c['id']=hashlib.sha256((str(i)+c['source']).encode()).hexdigest()[:12]
path=(P.parent if P.name=='sidewalk_data_3_support' else P)/'dino_sidewalk_data_3.ipynb';path.write_text(json.dumps(nb,indent=1)+'\n')
print('Built',path,'cells',len(cells),'bytes',path.stat().st_size)
