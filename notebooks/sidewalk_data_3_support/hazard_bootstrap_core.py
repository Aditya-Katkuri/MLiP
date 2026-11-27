"""Train 128 bootstrap hazard heads and export raw per-member probabilities."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, io, json, os, shutil, tempfile, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def plain(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)

def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), default=plain)

def fingerprint(x): return hashlib.sha256(canonical(x).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_bytes(payload); os.replace(tmp, path)

def save_json(path, value): atomic_bytes(path, (json.dumps(value, indent=2, default=plain, allow_nan=False)+'\n').encode())

def save_csv(path, frame): atomic_bytes(path, frame.to_csv(index=False).encode())

def save_torch(path, value):
    with tempfile.TemporaryDirectory() as temp:
        local = Path(temp)/'model.pt'; torch.save(value, local)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.'+path.name+'.tmp'); shutil.copyfile(local, tmp); os.replace(tmp, path)

def save_npz(path, **arrays):
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); atomic_bytes(path, buf.getvalue())

def batch_indices(n, steps, batch_size, generator, device):
    assert n>=batch_size
    order=torch.randperm(n,generator=generator); offset=0
    for _ in range(steps):
        if offset+batch_size>n: order=torch.randperm(n,generator=generator);offset=0
        yield order[offset:offset+batch_size].to(device)
        offset+=batch_size

def cache_features(encoder, df, split, root, run_dir, binding, batch_size, device):
    cache=Path(run_dir)/'features'/(split+'.pt')
    cache_binding={**binding,'split':split,'sample_ids_sha256':fingerprint(df.sample_id.tolist())}
    if cache.exists():
        cached=torch.load(cache,map_location='cpu',weights_only=True)
        assert cached['binding']==cache_binding,'Feature cache identity mismatch; start a new run.'
        x=cached['features']; assert x.shape==(len(df),encoder.config.hidden_size) and torch.isfinite(x).all()
        print('Reused cached features:',split)
        return x.to(device).clone()
    root=Path(root);mean=torch.tensor(MEAN,device=device).view(1,3,1,1);std=torch.tensor(STD,device=device).view(1,3,1,1)
    def read(row):
        payload=(root/row.image_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==row.output_file_sha256,f'Image checksum mismatch: {row.image_path}'
        with Image.open(io.BytesIO(payload)) as im:
            im.load();assert im.size==(224,224) and im.mode=='RGB' and im.format=='PNG'
            return np.array(im,copy=True)
    pieces=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for begin in range(0,len(df),batch_size):
            pixels=np.stack(list(pool.map(read,df.iloc[begin:begin+batch_size].itertuples(index=False))))
            x=torch.from_numpy(pixels).permute(0,3,1,2).to(device).float()/255
            x=(x-mean)/std
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
                feats=encoder(pixel_values=x).last_hidden_state[:,0].float()
            assert torch.isfinite(feats).all(),'Nonfinite backbone features; stop rather than saving a bad cache.'
            pieces.append(feats.cpu())
            if begin%(batch_size*20)==0:print(f'{split}: {min(begin+batch_size,len(df))}/{len(df)} embedded',flush=True)
    # clone outside inference_mode so head autograd can save its inputs.
    output=torch.cat(pieces).clone()
    save_torch(cache,dict(binding=cache_binding,features=output))
    print(f'Cached {split} in {time.perf_counter()-start:.1f}s')
    return output.to(device).clone()
def load_hazard_data(root, expected_hashes):
    root = Path(root)
    for filename, expected in expected_hashes.items():
        if file_sha(root / filename) != expected:
            raise ValueError(f'Dataset identity mismatch: {filename}')
    schema = json.loads((root / 'label_schema.json').read_text())
    assert schema['binary_labels'] == {'non_hazard': 0, 'hazard': 1}
    data = {}
    for split in ['pitt-train', 'pitt-val', 'pitt-test']:
        df = pd.read_csv(root / (split + '.csv'), keep_default_na=False, low_memory=False)
        assert df.sample_id.is_unique and df.image_path.is_unique
        assert set(df.split) == {split} and set(df.hazard_label) == {0, 1}
        assert np.array_equal(df['class'], np.where(df.hazard_label, 'hazard', 'non_hazard'))
        assert df.output_rgb_pixel_sha256.str.fullmatch('[0-9a-f]{64}').all()
        for relative in df.image_path:
            path = root / relative
            assert path.resolve().is_relative_to(root.resolve()) and path.is_file()
        data[split] = df
    assert sum(map(len, data.values())) == len(set().union(*(set(df.sample_id) for df in data.values())))
    return data, schema


def bootstrap_sample(df, seed):
    """Resample exact-pixel clusters within their binary-label composition strata."""
    assert set(df.split) == {'pitt-train'}, 'Bootstrap only the training split.'
    assert df.sample_id.is_unique and set(df.hazard_label) == {0, 1}
    labels = df.hazard_label.to_numpy(dtype=np.int64)
    clusters = {}
    for i, key in enumerate(df.output_rgb_pixel_sha256.astype(str)):
        assert key
        clusters.setdefault(key, []).append(i)
    strata = {}
    for key, indices in sorted(clusters.items()):
        composition = (int((labels[indices] == 0).sum()), int((labels[indices] == 1).sum()))
        strata.setdefault(composition, []).append(key)
    rng = np.random.default_rng(seed)
    draws = []
    for composition, keys in sorted(strata.items()):
        for key in rng.choice(keys, size=len(keys), replace=True):
            draws.extend(clusters[key])
    draws = rng.permutation(np.asarray(draws, dtype=np.int64))
    counts = np.bincount(draws, minlength=len(df))
    assert len(draws) == len(df)
    assert np.array_equal(np.bincount(labels[draws], minlength=2), np.bincount(labels, minlength=2))
    for indices in clusters.values():
        assert len(set(counts[indices])) == 1
    return draws, counts


def bootstrap_frames(df, draws, counts):
    cols = ['sample_id', 'image_path', 'hazard_label', 'output_rgb_pixel_sha256']
    selected = df.iloc[draws][cols].reset_index(drop=True)
    selected.insert(0, 'draw_position', np.arange(len(draws)))
    selected.insert(1, 'original_row_index', draws)
    membership = df[cols].copy()
    membership.insert(0, 'original_row_index', np.arange(len(df)))
    membership['in_bag_count'] = counts
    membership['out_of_bag'] = counts == 0
    return selected, membership


def member_ids(entries):
    ids = np.asarray([entry['cfg']['member'] for entry in entries], dtype=np.int64)
    assert np.array_equal(ids, np.arange(1, len(entries)+1)), 'Members must be ordered and contiguous.'
    return ids


def probability_columns(ids):
    return [f'p_hazard_head_{int(member):03d}' for member in ids]


@torch.inference_mode()
def predict_member(state, features):
    head = nn.Linear(features.shape[1], 1).to(features.device)
    head.load_state_dict(state)
    return head.eval()(features).sigmoid().float().cpu().numpy()[:, 0]


@torch.inference_mode()
def predict_ensemble(entries, features):
    member_ids(entries)
    weight = torch.cat([entry['state_dict']['weight'] for entry in entries]).to(features.device)
    bias = torch.cat([entry['state_dict']['bias'] for entry in entries]).to(features.device)
    p = (features @ weight.T + bias).sigmoid().float().cpu().numpy()
    assert p.shape == (len(features), len(entries)) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def train_member(cfg, train_features, train_df, draws, settings):
    """Only training features/labels enter optimization; retain the final update."""
    assert set(train_df.split) == {'pitt-train'}
    device = train_features.device
    torch.manual_seed(cfg['init_seed'])
    generator = torch.Generator().manual_seed(cfg['batch_seed'])
    head = nn.Linear(train_features.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    x = train_features[torch.as_tensor(draws, device=device)]
    y = torch.as_tensor(train_df.hazard_label.to_numpy()[draws], device=device, dtype=torch.float32)
    history, window = [], []
    for step, idx in enumerate(batch_indices(len(x), settings['updates'], settings['batch_size'], generator, device), 1):
        loss = nn.functional.binary_cross_entropy_with_logits(head(x[idx])[:, 0], y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        window.append(loss.detach())
        if step % settings['log_every'] == 0 or step == settings['updates']:
            mean_loss = float(torch.stack(window).mean())
            assert np.isfinite(mean_loss), 'Nonfinite training loss.'
            history.append(dict(member=cfg['member'], step=step, train_bce=mean_loss))
            window = []
    state = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
    assert all(torch.isfinite(tensor).all() for tensor in state.values())
    return state, pd.DataFrame(history)


def run_ensemble(settings, binding, train_features, train_df, run_dir):
    assert settings['n_members'] == 128, 'This experiment trains exactly 128 heads.'
    assert set(train_df.split) == {'pitt-train'} and len(train_features) == len(train_df)
    entries, histories, summaries = [], [], []
    for member in range(1, settings['n_members']+1):
        cfg = dict(member=member, bootstrap_seed=settings['bootstrap_seed_base']+member,
                   init_seed=settings['init_seed_base']+member, batch_seed=settings['batch_seed_base']+member)
        draws, counts = bootstrap_sample(train_df, cfg['bootstrap_seed'])
        folder = Path(run_dir)/'members'/f'head_{member:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        done = folder/'complete.json'
        identity = dict(cfg=cfg, settings=settings, binding=binding,
                        train_sample_ids_sha256=fingerprint(train_df.sample_id.tolist()),
                        bootstrap_draws_sha256=fingerprint(draws))
        if done.exists():
            completion = json.loads(done.read_text())
            assert canonical(completion['identity']) == canonical(identity), 'Resume identity mismatch.'
            for name, expected in completion['file_hashes'].items():
                assert file_sha(folder/name) == expected, f'Damaged member artifact: {name}'
            payload = torch.load(folder/'head.pt', map_location='cpu', weights_only=True)
            assert canonical(payload['identity']) == canonical(identity)
            state, history = payload['state_dict'], pd.read_csv(folder/'history.csv')
            action = 'reused'
        else:
            selected, membership = bootstrap_frames(train_df, draws, counts)
            save_csv(folder/'bootstrap_draws.csv', selected)
            save_csv(folder/'bootstrap_counts.csv', membership)
            start = time.perf_counter()
            state, history = train_member(cfg, train_features, train_df, draws, settings)
            save_torch(folder/'head.pt', dict(format_version=2, task='hazard', identity=identity,
                       feature_dim=train_features.shape[1], output_labels=['hazard'], state_dict=state))
            reloaded = torch.load(folder/'head.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(state[k], reloaded['state_dict'][k]) for k in state)
            save_csv(folder/'history.csv', history)
            save_json(folder/'timing.json', dict(training_and_save_seconds=time.perf_counter()-start))
            files = ['head.pt', 'bootstrap_draws.csv', 'bootstrap_counts.csv', 'history.csv', 'timing.json']
            save_json(done, dict(identity=identity, file_hashes={name:file_sha(folder/name) for name in files}))
            action = 'trained'
        entries.append(dict(cfg=cfg, state_dict=state, counts=counts))
        histories.append(history)
        summaries.append(dict(**cfg, bootstrap_rows=len(draws), unique_rows=int((counts>0).sum()),
                              out_of_bag_rows=int((counts==0).sum()), hazards=int(train_df.hazard_label.to_numpy()[draws].sum()),
                              bootstrap_draws_sha256=fingerprint(draws)))
        save_json(Path(run_dir)/'status.json', dict(status='training', members_completed=member, members_total=settings['n_members']))
        print(f"[{member}/{settings['n_members']}] {action}; {settings['updates']} updates; {int((counts>0).sum())} unique training records", flush=True)
    assert len({s['bootstrap_draws_sha256'] for s in summaries}) == settings['n_members']
    history = pd.concat(histories, ignore_index=True)
    save_csv(Path(run_dir)/'training_history.csv', history)
    save_csv(Path(run_dir)/'bootstrap_members.csv', pd.DataFrame(summaries))
    save_npz(Path(run_dir)/'pitt-train_bootstrap_counts.npz',
             counts=np.column_stack([entry['counts'] for entry in entries]),
             sample_ids=train_df.sample_id.to_numpy(dtype=str), member_ids=member_ids(entries))
    return entries


def export_predictions(entries, features, df, run_dir, split):
    assert set(df.split) == {split} and len(df) == len(features)
    ids = member_ids(entries)
    p = predict_ensemble(entries, features)
    metadata_columns = ['sample_id', 'image_path', 'split', 'hazard_label', 'original_class',
                        'city', 'is_manual_collection', 'output_file_sha256', 'output_rgb_pixel_sha256']
    frame = pd.concat([df[[c for c in metadata_columns if c in df]].reset_index(drop=True),
                       pd.DataFrame(p, columns=probability_columns(ids))], axis=1)
    save_csv(Path(run_dir)/(split+'_predictions.csv'), frame)
    save_npz(Path(run_dir)/(split+'_probabilities.npz'), probabilities=p,
             sample_ids=df.sample_id.to_numpy(dtype=str), member_ids=ids)
    return frame


def export_ensemble(entries, binding, check_features, run_dir):
    root = Path(run_dir)/'inference'
    root.mkdir(exist_ok=True)
    ids = member_ids(entries)
    payload = dict(format_version=2, task='hazard', n_members=len(entries), feature_dim=check_features.shape[1],
                   binding=binding, member_configs=[e['cfg'] for e in entries], member_ids=ids.tolist(),
                   probability_columns=probability_columns(ids),
                   weight=torch.cat([e['state_dict']['weight'] for e in entries]),
                   bias=torch.cat([e['state_dict']['bias'] for e in entries]))
    if (root/'ensemble.pt').exists():
        previous = torch.load(root/'ensemble.pt', map_location='cpu', weights_only=True)
        for key in ['format_version', 'task', 'n_members', 'feature_dim', 'binding', 'member_configs', 'member_ids', 'probability_columns']:
            assert canonical(previous[key]) == canonical(payload[key]), 'Export identity changed.'
        assert torch.equal(previous['weight'], payload['weight']) and torch.equal(previous['bias'], payload['bias'])
    else:
        save_torch(root/'ensemble.pt', payload)
    saved = torch.load(root/'ensemble.pt', map_location=check_features.device, weights_only=True)
    with torch.inference_mode():
        p = (check_features @ saved['weight'].T + saved['bias']).sigmoid().cpu().numpy()
    expected = np.column_stack([predict_member(e['state_dict'], check_features) for e in entries])
    error = float(np.max(np.abs(p-expected)))
    assert error < 2e-6
    save_json(root/'checkpoint_reload_verification.json', dict(passed=True, max_probability_difference=error, n_members=len(entries)))
    return root
