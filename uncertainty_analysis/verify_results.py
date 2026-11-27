"""Audit saved artifacts against raw ensemble probabilities and source labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import Dataset, SPLITS, parser, pairwise_counts, rank_metrics, save_json, sha256
from .consensus_solver import agreement
from .point_2_adjusted_scores import select_lambda


def main():
    args = parser(__doc__).parse_args()
    source = Dataset(args.archive)
    audit = {}
    method_sets = {'point_2': ['mean', 'adjusted', 'lower_q05'],
                   'point_3': ['mean', 'expected_rank', 'top_k_frequency'],
                   'point_4': ['mean', 'expected_rank', 'consensus']}
    for split in SPLITS:
        data = source.load(split)
        ids = pd.Index(data.ids)
        wins, ties = pairwise_counts(data.p)
        audits = {}
        for point, names in method_sets.items():
            folder = args.output / point
            metrics = json.loads((folder/'metrics.json').read_text())['metrics'][split]
            for name in names:
                frame = pd.read_csv(folder/f'{split}_{name}.csv')
                order = ids.get_indexer(frame.sample_id)
                assert (order >= 0).all() and len(np.unique(order)) == len(ids)
                assert frame['rank'].tolist() == list(range(1,len(ids)+1))
                assert np.array_equal(frame.hazard_label, data.y[order])
                assert np.allclose(frame.p_hazard_mean, data.p.mean(1)[order], rtol=1e-12, atol=1e-12)
                assert np.allclose(frame.p_hazard_sd, data.p.std(1,ddof=1)[order], rtol=1e-12, atol=1e-12)
                recomputed = rank_metrics(data.y, order, args.k)
                assert recomputed == metrics[name]
                subset = pd.read_csv(folder/f'{split}_{name}_top{args.k}.csv')
                pd.testing.assert_frame_equal(subset, frame.iloc[:args.k].reset_index(drop=True))
                audits[f'{point}/{name}'] = recomputed
            if point in ['point_3','point_4']:
                with np.load(folder/f'{split}_pairwise.npz') as z:
                    assert np.array_equal(z['sample_ids'],data.ids)
                    assert np.array_equal(z['wins'],wins) and np.array_equal(z['ties'],ties)
                    assert np.allclose(z['p_strict'],wins/128)
                    assert np.allclose(z['p_tie'],ties/128)
                matrix = pd.read_csv(folder/f'{split}_pairwise_probability.csv',index_col=0)
                assert matrix.index.tolist()==data.ids.tolist()==matrix.columns.tolist()
                assert np.allclose(matrix,wins/128)
        rank_frame = pd.read_csv(args.output/'point_3'/f'{split}_expected_rank.csv')
        assert np.isclose(rank_frame.p_first.sum(),1) and np.isclose(rank_frame.p_top_k.sum(),args.k)
        stable = rank_frame.p_top_k >= .95
        head_hits = []
        for b in range(data.p.shape[1]):
            order = np.lexsort((data.ids, -data.p[:,b]))
            head_hits.append(int(data.y[order[:args.k]].sum()))
        freq = pd.read_csv(args.output/'point_3'/f'{split}_top_k_frequency.csv')
        mean = pd.read_csv(args.output/'point_3'/f'{split}_mean.csv')
        freq_entered = freq.head(args.k)[~freq.head(args.k).sample_id.isin(mean.head(args.k).sample_id)]
        freq_exited = mean.head(args.k)[~mean.head(args.k).sample_id.isin(freq.head(args.k).sample_id)]
        cert = json.loads((args.output/'point_4'/f'{split}_solver.json').read_text())
        consensus_frame = pd.read_csv(args.output/'point_4'/f'{split}_consensus.csv')
        order = ids.get_indexer(consensus_frame.sample_id)
        assert agreement(wins, order) == cert['agreement']
        assert cert['upper_bound'] >= cert['agreement'] and cert['absolute_gap'] == cert['upper_bound']-cert['agreement']
        if cert['optimal']:
            assert cert['absolute_gap'] == 0
        pixels = data.frame.groupby('output_rgb_pixel_sha256').hazard_label.agg(['size','nunique'])
        audit[split] = dict(n_images=len(data.ids), hazards=int(data.y.sum()),
            verified_rankings=len(audits), metrics=audits,
            duplicate_pixel_groups=int((pixels['size']>1).sum()),
            conflicting_pixel_groups=int((pixels['nunique']>1).sum()),
            stable_top_k_count=int(stable.sum()), stable_top_k_hazards=int(rank_frame.loc[stable,'hazard_label'].sum()),
            head_precision_range=[min(head_hits)/args.k,max(head_hits)/args.k],
            head_precision_mean=float(np.mean(head_hits)/args.k),
            frequency_vs_mean_entered=freq_entered[['sample_id','hazard_label','p_top_k']].to_dict('records'),
            frequency_vs_mean_exited=freq_exited[['sample_id','hazard_label','p_top_k']].to_dict('records'),
            frequency_cutoff_tied=bool(freq.p_top_k.iloc[args.k-1]==freq.p_top_k.iloc[args.k]),
            mean_cutoff_tied=bool(mean.p_hazard_mean.iloc[args.k-1]==mean.p_hazard_mean.iloc[args.k]),
            consensus_optimal=cert['optimal'],consensus_objective=cert['agreement'])
    val = source.load('pitt-val')
    selected, _ = select_lambda(val.p,val.y,val.ids,args.k)
    assert selected == json.loads((args.output/'point_2/selection.json').read_text())['lambda_value']
    artifacts={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=sha256(p))
               for p in sorted(args.output.rglob('*')) if p.is_file() and p.name not in ['verification.json','artifact_manifest.json']}
    save_json(args.output/'artifact_manifest.json',artifacts)
    save_json(args.output/'verification.json',dict(status='passed',k=args.k,selection_recomputed_from_validation_only=selected,
                                                verified_artifact_count=len(artifacts),splits=audit))
    print(json.dumps(dict(status='passed',splits=audit),indent=2))


if __name__ == '__main__':
    main()
