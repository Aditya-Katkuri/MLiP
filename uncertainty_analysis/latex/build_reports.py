"""Generate three standalone LaTeX reports from the verified saved analysis."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
FIGURES = HERE / 'figures'
BUILD = HERE / 'build'
FIGURES.mkdir(exist_ok=True)
BUILD.mkdir(exist_ok=True)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value, digits=2):
    return f'{100 * value:.{digits}f}'


p2 = read_json(RESULTS / 'point_2/metrics.json')
p3 = read_json(RESULTS / 'point_3/metrics.json')
p4 = read_json(RESULTS / 'point_4/metrics.json')
audit = read_json(RESULTS / 'verification.json')
certificates = {s: read_json(RESULTS / 'point_4' / f'{s}_solver.json') for s in ['pitt-val', 'pitt-test']}
assert audit['status'] == 'passed'
assert p2['selection']['k'] == p3['k'] == p4['k'] == 46
assert p2['selection']['lambda_value'] == 0
assert all(c['optimal'] for c in certificates.values())

# These documents describe the archived experiment. Stop rather than silently reuse
# narrative conclusions when their supporting results change.
expected_hits = {
    'point_2': {'mean': (34, 27), 'adjusted': (34, 27), 'lower_q05': (32, 26)},
    'point_3': {'mean': (34, 27), 'expected_rank': (34, 27), 'top_k_frequency': (34, 28)},
    'point_4': {'mean': (34, 27), 'expected_rank': (34, 27), 'consensus': (34, 27)},
}
for name, report in [('point_2', p2), ('point_3', p3), ('point_4', p4)]:
    for method, hits in expected_hits[name].items():
        assert tuple(report['metrics'][s][method]['hits'] for s in ['pitt-val', 'pitt-test']) == hits

PREAMBLE = r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=25mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage{caption,float}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\newcommand{\ind}{\mathbf{1}}
\newcommand{\Ptop}{\widehat{P}_{46}}
\begin{document}
'''

SETUP = r'''\section{Evaluation setup}
We analyze hazard probabilities from $B=128$ separately fitted linear heads over
frozen DINOv3 ViT-B/16 features. Each head was trained on a bootstrap sample of
1,276 Pittsburgh training observations (527 hazards and 749 non-hazards), with
exact-pixel clusters sampled within strata defined by their binary-label composition.
All heads used Adam, learning rate $3\times10^{-5}$, weight decay $0.01$, batch
size 64, 5,000 updates, unweighted binary cross-entropy, and the final checkpoint.

The validation set contains 382 observations, including 157 hazards; the test set
contains 380 observations, including 174 hazards. The review budget is fixed at
$k=46$: 12.04\% of validation and 12.11\% of test. For an ordering $\pi$, the metrics are
\begin{equation}
\operatorname{Precision@46}=\frac{1}{46}\sum_{a=1}^{46}y_{\pi(a)},
\qquad
\operatorname{Recall@46}=\frac{\sum_{a=1}^{46}y_{\pi(a)}}{\sum_{i=1}^{N}y_i},
\end{equation}
where $y_i=1$ denotes a hazard. Subtype predictions are excluded.
'''

LIMITS = r'''The 128 heads share a frozen representation and overlapping bootstrap training
samples. Their variation measures sensitivity to these model refits, not all
sources of predictive uncertainty. Empirical intervals are not automatically
calibrated posterior intervals or confidence intervals for population precision.
Original evaluation observations are retained, including duplicate pixel groups
and groups with conflicting hazard labels: 18 duplicate groups, of which 6 have
conflicting labels, in validation; 12 and 2, respectively, in test. Shared confident
mistakes and label conflicts can therefore persist across the ensemble.
'''


def retrieval_table(report, methods, label):
    rows = []
    for split, title in [('pitt-val', 'Validation'), ('pitt-test', 'Test')]:
        for i, (method, name) in enumerate(methods):
            m = report['metrics'][split][method]
            rows.append(f"{title if i == 0 else ''} & {name} & {m['hits']}/46 & "
                        f"{percent(m['precision_at_k'])} & {percent(m['recall_at_k'])} \\\\")
        if split == 'pitt-val':
            rows.append(r'\midrule')
    return (r'''\begin{table}[H]
\centering
\caption{Hazard retrieval at the fixed budget of 46 images. Precision and recall are percentages.}
\label{''' + label + r'''}
\small
\begin{tabular}{llrrr}
\toprule
Split & Ranking & Hazards & Precision@46 & Recall@46 \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def copy_figure(point, name):
    source = RESULTS / point / (name + '.pdf')
    dest = FIGURES / (point + '_' + name + '.pdf')
    shutil.copy2(source, dest)
    return 'figures/' + dest.name


def figure(path, caption, label, width='.9', placement='htbp'):
    return (r'\begin{figure}[' + placement + ']\n' + r'\centering' + '\n' +
            r'\includegraphics[width=' + width + r'\linewidth]{' + path + '}\n' +
            r'\caption{' + caption + '}\n' + r'\label{' + label + '}\n' + r'\end{figure}' + '\n')


def document(title, body):
    return PREAMBLE + r'\hypersetup{pdftitle={' + title + '}}\n' + r'\section*{' + title + '}\n' + body + '\n' + r'\end{document}' + '\n'


point2 = SETUP + r'''
\section{Uncertainty-adjusted scores}
Let $p_i^{(b)}$ be head $b$'s hazard probability for observation $i$. Define
\begin{equation}
\mu_i=\frac{1}{B}\sum_{b=1}^{B}p_i^{(b)},\qquad
s_i=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(p_i^{(b)}-\mu_i)^2},\qquad
S_i(\lambda)=\mu_i-\lambda s_i.
\end{equation}
Images are ranked by decreasing $S_i$. Here $s_i$ is the sample standard deviation
across heads, not the standard error of their average. Scores are not clipped to
$[0,1]$; exact score ties are resolved by ascending sample ID.

We search 81 prespecified values $\lambda\in\{0,0.05,\ldots,4\}$, maximizing
validation precision@46. Ties are resolved by the smallest $\lambda$. The selected
value is frozen before test evaluation. Mean-probability ranking ($\lambda=0$)
and a fixed fifth-percentile probability score are comparators; the latter does
not participate in selection. No parameter is selected using test labels.

\section{Results}
The best validation result is 34 hazards among 46 selected images
(73.91\% precision). Eight grid values, $\lambda=0$ through $0.35$, tie for this
result. The tie rule therefore selects $\lambda=0$, so the selected method is
exactly mean-probability ranking. All remaining grid values ($0.40$--$4.00$)
retrieve 33 validation hazards (71.74\%). This is selection within the searched
grid, not a claim about every real-valued penalty.
'''
point2 += retrieval_table(p2, [('mean', 'Mean probability'), ('adjusted', r'Selected $\mu-\lambda s$ ($\lambda=0$)'),
                              ('lower_q05', 'Fifth probability percentile')], 'tab:adjusted')
point2 += figure(copy_figure('point_2', 'validation_lambda_sweep'),
                 r'Validation precision@46 over the uncertainty-penalty grid. The smallest maximizer, $\lambda=0$, is selected.',
                 'fig:lambda', width='.87')
point2 += r'''
\section{Interpretation}
The selected ranking retrieves 27 test hazards (58.70\% precision@46;
15.52\% recall@46). The fixed fifth-percentile comparator retrieves 26
(56.52\% precision; 14.94\% recall). Thus, neither conservative scoring approach
improves retrieval here: the validation-selected SD penalty is zero, and the
lower-quantile comparator performs worse on both splits.

If the mean probabilities equal the relevant predictive expectations, choosing
the largest means maximizes the expected number of hazards at a fixed review
budget. An SD penalty instead expresses a preference for stability; it need not
increase hazard yield. Each extra correct retrieval changes precision@46 by
2.17 percentage points. The validation optimum was used for selection and is
therefore optimistic; these results do not establish a population-level or
statistically significant improvement.

\section{Limitations}
''' + LIMITS

point3 = SETUP + r'''
\section{Rank distributions and pairwise probabilities}
For each head, rank all observations together by decreasing hazard probability,
with rank 1 the highest. Let $r_i^{(b)}$ be the resulting rank, using midranks for
exact probability ties. We summarize
\begin{equation}
\overline{r}_i=\frac{1}{B}\sum_{b=1}^{B}r_i^{(b)},\qquad
I_i=\left[Q_{0.025}(r_i^{(1:B)}),\ Q_{0.975}(r_i^{(1:B)})\right].
\end{equation}
Quantiles use linear interpolation. These are intervals of midranks across
refits; they do not include hypothetical random tie-breaking noise.

For selection frequencies, tied slots are allocated equally. If head $b$ assigns
$a_i^{(b)}$ observations strictly higher scores and $t_i^{(b)}$ observations the
same score (including $i$), define
\begin{equation}
\widehat{P}_{k}(i)=\frac{1}{B}\sum_{b=1}^{B}
\min\!\left\{1,\max\!\left\{0,\frac{k-a_i^{(b)}}{t_i^{(b)}}\right\}\right\}.
\end{equation}
We report $\widehat{P}_1(i)$ and $\widehat{P}_{46}(i)$. Their sums over images are
1 and 46, respectively. These frequencies quantify selection stability across
heads, not the probability that an image is a true hazard.

For each image pair, compare predictions from the same head:
\begin{equation}
W_{ij}=\frac{1}{B}\sum_{b=1}^{B}\ind\!\left[p_i^{(b)}>p_j^{(b)}\right],\qquad
T_{ij}=\frac{1}{B}\sum_{b=1}^{B}\ind\!\left[p_i^{(b)}=p_j^{(b)}\right].
\end{equation}
Then $W_{ij}+W_{ji}+T_{ij}=1$, $W_{ii}=0$ and $T_{ii}=1$. Matching head indices
preserves dependence between predictions. In particular,
\begin{equation}
\overline{r}_i=1+\sum_{j\ne i}\left(W_{ji}+\tfrac12 T_{ij}\right),
\end{equation}
so sorting by increasing expected rank is equivalent to weighted Borda ordering.
We compare this primary ordering with mean probability and decreasing
$\widehat{P}_{46}$. Ties in selection frequency are broken by expected rank, then
sample ID; equal expected ranks use sample ID. No labels determine these orderings.

\section{Retrieval results}
'''
point3 += retrieval_table(p3, [('mean', 'Mean probability'), ('expected_rank', 'Expected rank'),
                              ('top_k_frequency', 'Top-46 frequency')], 'tab:ranks')
point3 += r'''
All three methods retrieve 34 validation hazards (73.91\%). Mean and expected-rank
orderings retrieve 27 test hazards (58.70\%); top-46 frequency retrieves 28
(60.87\%). The frequency ordering replaces one selected test non-hazard with a
hazard, a 2.17 percentage-point difference. Neither the mean nor the frequency
ordering has a tie at its test selection boundary. The validation score does not
distinguish the methods, and the test difference is not used to select a winner.

\section{Rank stability}
'''
uval, utest = p3['uncertainty']['pitt-val'], p3['uncertainty']['pitt-test']
stability_rows = []
for key, name, decimal in [
    ('median_interval_width', 'Median rank-interval width', True),
    ('p90_interval_width', '90th-percentile interval width', True),
    ('intervals_crossing_cutoff', 'Intervals crossing rank 46', False),
    ('observations_p_top_k_ge_095', r'Images with $\widehat{P}_{46}\geq0.95$', False),
    ('hazards_among_stable_top_k', 'Hazards among these stable selections', False),
    ('nonhazards_among_stable_top_k', 'Non-hazards among these stable selections', False),
    ('observations_p_top_k_positive', r'Images with $\widehat{P}_{46}>0$', False),
]:
    a, b = uval[key], utest[key]
    stability_rows.append(f'{name} & '+(f'{a:.1f} & {b:.1f}' if decimal else f'{a} & {b}')+r' \\')
point3 += r'''\begin{table}[H]
\centering
\caption{Stability across 128 heads. Interval widths are measured in rank positions; other rows count observations.}
\label{tab:stability}
\small
\begin{tabular}{lrr}
\toprule
Statistic & Validation & Test \\
\midrule
''' + '\n'.join(stability_rows) + r'''
\bottomrule
\end{tabular}
\end{table}

The test median rank-interval width is 162.2 positions, and 112 intervals cross
the review cutoff. Only 14 test images appear in the top 46 in at least 95\% of
heads. Of these, 8 are labelled hazards and 6 non-hazards. Thus, stability can
coexist with systematic false-positive selections. Mean and expected-rank
orderings nevertheless select exactly the same 46 images on each split: agreement
between their aggregate selections does not imply narrow per-image rank intervals.

\section{Interpretation and limitations}
Rank intervals show where placement is sensitive to refitting, while top-46
frequency identifies stable review candidates. These quantities can accompany a
ranking without changing its ordering. The one-image test gain from frequency
ranking is descriptive and does not establish general superiority or statistical
significance. The pairwise probabilities also quantify whether any particular
comparison is well supported by the ensemble, rather than forcing every ordering
to appear equally certain.

''' + LIMITS + r'''
\clearpage
\section{Rank-uncertainty figures}
'''
point3 += figure(copy_figure('point_3', 'pitt-val_rank_intervals'),
                 r'Validation rank intervals. Observations are ordered by expected rank; shading shows the 2.5th--97.5th percentile interval and the dashed line marks rank 46.',
                 'fig:val-rank', width='.9', placement='H')
point3 += figure(copy_figure('point_3', 'pitt-test_rank_intervals'),
                 r'Test rank intervals, with the same definition. Smaller ranks are better, so the vertical axis increases downwards.',
                 'fig:test-rank', width='.9', placement='H')
point3 += figure(copy_figure('point_3', 'pitt-test_top_k_frequency'),
                 r'Test top-46 selection frequencies, plotted in expected-rank order. The vertical line marks the first 46 positions of that aggregate ordering.',
                 'fig:test-frequency', width='.9', placement='H')

point4 = SETUP + r'''
\section{Pairwise outranking and the consensus objective}
For each ordered pair of observations, estimate the strict outranking frequency
\begin{equation}
W_{ij}=\frac{1}{B}\sum_{b=1}^{B}\ind\!\left[p_i^{(b)}>p_j^{(b)}\right],
\qquad C_{ij}=B W_{ij}.
\end{equation}
Comparisons use the same head for both observations. Let $T_{ij}$ denote the
fraction of exact score ties; then $W_{ij}+W_{ji}+T_{ij}=1$. A pair tied in every
head has $W_{ij}=W_{ji}=0$, not evidence of mutual losses. There are 22 such
unordered pairs in validation and 12 in test. The strict diagonal is zero.

The consensus ordering solves the Kemeny objective
\begin{equation}
\pi^*\in\operatorname*{arg\,max}_{\pi}
\sum_{1\leq a<b\leq N}C_{\pi(a),\pi(b)}.
\label{eq:kemeny}
\end{equation}
It maximizes agreement between the final ordering and the heads' pairwise
preferences. Adding half-credit for score ties adds the same constant to every
permutation, so it leaves the optimum unchanged. This objective differs from
both expected-rank/Borda ordering and maximizing the number of true hazards in
the top 46. No hazard labels enter the optimizer.

\section{Optimization and certification}
Pairwise majority preferences can form cycles. We first decompose the majority
graph into strongly connected components, using edges in both directions when
$C_{ij}=C_{ji}$. Cross-component preferences then form an acyclic order. Within
each component, expected-rank ordering and improving insertions provide a
feasible starting solution.

A binary linear-ordering program defines $x_{ij}=1$ when $i$ precedes $j$, for
$i<j$. The transitivity constraints are
\begin{equation}
0\leq x_{ij}+x_{jk}-x_{ik}\leq1,\qquad i<j<k.
\end{equation}
Violated triangle constraints are added iteratively. A feasible ordering and an
upper bound from the mixed-integer relaxation are maintained. Matching the bound
certifies a global optimum; because agreement counts are integers, a remaining
gap strictly below one also suffices. Both splits reached a certified optimum.
'''
cert_rows = []
for s, title in [('pitt-val', 'Validation'), ('pitt-test', 'Test')]:
    c = certificates[s]
    cert_rows.append(f"{title} & {c['n_components']} & {max(c['component_sizes'])} & "
                     f"{c['agreement']:,} & {c['upper_bound']:,.0f} & {c['absolute_gap']:.0f} \\\\")
point4 += r'''\begin{table}[H]
\centering
\caption{Optimality certificates. Objective and bound are integer head-pair agreement counts, not hazard counts.}
\label{tab:certificate}
\small
\begin{tabular}{lrrrrr}
\toprule
Split & Components & Largest size & Objective & Upper bound & Gap \\
\midrule
''' + '\n'.join(cert_rows) + r'''
\bottomrule
\end{tabular}
\end{table}

For comparison across splits, normalize agreement by $B\binom{N}{2}$ and award
half-credit to score ties:
\begin{equation}
A(\pi)=\frac{\sum_{a<b}C_{\pi(a),\pi(b)}+\tfrac12 B\sum_{i<j}T_{ij}}
{B\binom{N}{2}}.
\end{equation}
'''
agreement_rows = []
for s, title in [('pitt-val', 'Validation'), ('pitt-test', 'Test')]:
    c = certificates[s]
    baseline = (c['initial_agreement'] + c['score_tie_half_credit_constant']) / c['head_pair_comparisons']
    consensus = c['agreement_fraction_half_ties']
    agreement_rows.append(f"{title} & {percent(baseline,4)} & {percent(consensus,4)} & "
                          f"{100*(consensus-baseline):.4f} \\\\")
point4 += r'''\begin{table}[H]
\centering
\caption{Pairwise agreement (\%) with half-credit for score ties. The gain is in percentage points.}
\label{tab:agreement}
\small
\begin{tabular}{lrrr}
\toprule
Split & Expected-rank/Borda & Optimal consensus & Gain \\
\midrule
''' + '\n'.join(agreement_rows) + r'''
\bottomrule
\end{tabular}
\end{table}

\section{Hazard retrieval and interpretation}
'''
point4 += retrieval_table(p4, [('mean', 'Mean probability'), ('expected_rank', 'Expected rank'),
                              ('consensus', 'Optimal pairwise consensus')], 'tab:consensus-retrieval')
point4 += r'''
Despite improving the pairwise-agreement objective, consensus retrieves the same
number of hazards as mean and expected-rank ordering: 34 on validation and 27 on
test. Relative to expected-rank ordering, 277 validation positions and 279 test
positions change. The selected sets still overlap in 46 of 46 validation images
and 45 of 46 test images. The test replacement does not change the hazard count.

A certified pairwise optimum therefore does not imply improved hazard retrieval.
Global agreement gives equal weight to all image pairs; it can be high even when
ordering near the review boundary is uncertain. Most pairs are far apart in rank,
whereas precision@46 depends only on the selected set. Multiple equally optimal
permutations may exist. The consensus is best interpreted alongside per-image
rank intervals and top-46 frequencies.

\section{Limitations}
''' + LIMITS + r'''
The optimality certificates concern these stored predictions and the objective
in Equation~\ref{eq:kemeny}; they are not guarantees about true hazard labels or
new images. Bootstrap outranking frequencies describe variation across fitted
heads rather than literal Bayesian posterior probabilities.

\clearpage
\section{Pairwise probability heatmaps}
Rows and columns use the same consensus order. A cell gives the frequency with
which its row image has a strictly larger hazard score than its column image.
The color scale is $[0,1]$; diagonals are masked and dashed lines mark the top-46
boundary. Exact ties are retained separately, so opposite cells need not sum to
one. The accompanying numeric matrices retain every image identifier.
'''
point4 += figure(copy_figure('point_4', 'pitt-val_pairwise_heatmap'),
                 r'Validation pairwise outranking frequencies for all 382 observations, ordered by the certified consensus.',
                 'fig:val-pairwise', width='.88', placement='H')
point4 += r'\clearpage' + '\n'
point4 += figure(copy_figure('point_4', 'pitt-test_pairwise_heatmap'),
                 r'Test pairwise outranking frequencies for all 380 observations, ordered by the certified consensus.',
                 'fig:test-pairwise', width='.88', placement='H')
point4 += r'\clearpage' + '\n'
point4 += figure(copy_figure('point_4', 'pitt-test_top20_heatmap'),
                 r'Test top-20 consensus positions. Values are strict outranking frequencies rounded to two decimals. The zero-valued opposite cells at positions 4/5, 7/8 and 11/12 correspond to exact score ties across all heads.',
                 'fig:test-top20', width='1', placement='H')

reports = {
    'point_2_adjusted_scores': ('Point 2: Uncertainty-adjusted hazard ranking', point2),
    'point_3_rank_uncertainty': ('Point 3: Uncertainty in hazard ranks', point3),
    'point_4_pairwise_consensus': ('Point 4: Pairwise outranking and consensus', point4),
}

for stem, (title, body) in reports.items():
    tex = HERE / (stem + '.tex')
    tex.write_text(document(title, body))
    for pass_number in [1, 2]:
        command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
                   '-output-directory=build', tex.name]
        result = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
        (BUILD / f'{stem}.pass{pass_number}.txt').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{stem}: compilation failed; see {BUILD / (stem + ".pass" + str(pass_number) + ".txt")}')
    shutil.copy2(BUILD / (stem + '.pdf'), HERE / (stem + '.pdf'))
    subprocess.run(['pdftotext', '-layout', str(HERE / (stem + '.pdf')), str(BUILD / (stem + '.txt'))], check=True)
    print('Compiled', stem + '.pdf', flush=True)

source_files = [RESULTS / 'verification.json']
for p in ['point_2', 'point_3', 'point_4']:
    source_files += [RESULTS / p / 'metrics.json', RESULTS / p / 'analysis.md']
source_files += [RESULTS / 'point_4' / f'{s}_solver.json' for s in certificates]
source_files += list(FIGURES.glob('*.pdf'))
manifest = dict(inputs={str(p.relative_to(HERE.parent)): sha256(p) for p in source_files},
                outputs={p.name:sha256(p) for p in sorted(HERE.glob('*')) if p.suffix in ['.tex','.pdf']},
                builder_sha256=sha256(Path(__file__)),
                compiler=subprocess.run(['pdflatex','--version'],capture_output=True,text=True).stdout.splitlines()[0])
(HERE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
