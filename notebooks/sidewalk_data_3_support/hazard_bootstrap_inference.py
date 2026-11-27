"""Offline export of raw hazard probabilities from all saved bootstrap heads."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import torch
from PIL import Image, ImageOps


class HazardEnsemble:
    def __init__(self, bundle, device=None):
        from transformers import AutoModel
        self.bundle = Path(bundle)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessing = json.loads((self.bundle/'preprocessing.json').read_text())
        saved = torch.load(self.bundle/'ensemble.pt', map_location=self.device, weights_only=True)
        assert saved['format_version'] == 2 and saved['task'] == 'hazard'
        self.n_members = saved['n_members']
        self.columns = saved['probability_columns']
        self.weight, self.bias = saved['weight'], saved['bias']
        assert self.weight.shape[0] == self.bias.shape[0] == len(self.columns) == self.n_members
        self.encoder = AutoModel.from_pretrained(self.bundle/'backbone', local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.mean = torch.tensor(self.preprocessing['mean'], device=self.device).view(1,3,1,1)
        self.std = torch.tensor(self.preprocessing['std'], device=self.device).view(1,3,1,1)
        if str(self.device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    @torch.inference_mode()
    def predict(self, paths, batch_size=64):
        paths = [Path(p) for p in paths]
        assert len(paths) and len({str(p) for p in paths}) == len(paths)
        assert batch_size > 0
        rows = []
        for start in range(0, len(paths), batch_size):
            chosen, pixels = paths[start:start+batch_size], []
            for path in chosen:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB').resize(tuple(self.preprocessing['size']), Image.Resampling.BICUBIC)
                    pixels.append(np.array(image, copy=True))
            x = torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).to(self.device).float()/255
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=str(self.device).startswith('cuda')):
                features = self.encoder(pixel_values=(x-self.mean)/self.std).last_hidden_state[:,0].float()
            p = (features @ self.weight.T + self.bias).sigmoid().cpu().numpy()
            assert p.shape == (len(chosen), self.n_members) and np.isfinite(p).all()
            for i, path in enumerate(chosen):
                row = dict(image_path=str(path))
                row.update({column:float(p[i,j]) for j,column in enumerate(self.columns)})
                rows.append(row)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('hazard_predictions.csv'))
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    predictor = HazardEnsemble(args.bundle, args.device)
    rows = predictor.predict(args.images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} images with {predictor.n_members} raw hazard probabilities: {args.output}')


if __name__ == '__main__':
    main()
