import argparse
import csv
import importlib.metadata as md
import json
from pathlib import Path
import time

import numpy as np
import zarr
from cellpose import metrics, models
from ome_zarr.writer import write_labels
from tqdm import tqdm

from benchmark_utils import hardware_record, model_digest, normalise, read_zarr_image, read_zarr_labels, synchronize

THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)
DIAMETERS = [12, 17, 22, 27, 32]


def mean_ap(gt, pred):
    if gt.max() == 0:
        return 1.0 if pred.max() == 0 else 0.0
    if pred.max() == 0:
        return 0.0
    value = float(metrics.average_precision(gt, pred, threshold=THRESHOLDS)[0].mean())
    assert np.isfinite(value), 'Non-finite tuning score'
    return value


def choose_diameter(model, rows, results):
    records = []
    training = [r for r in rows if r['split'] == 'training']
    validation = [r for r in rows if r['split'] == 'validation']
    assert len(training) == 100 and len(validation) == 50
    training_cache = [(r, read_zarr_image(r['ome_zarr']), read_zarr_labels(r['ome_zarr'], 'ground_truth'))
                      for r in training]
    means = {}
    for diameter in DIAMETERS:
        values = []
        for row, img, gt in tqdm(training_cache, desc=f'nuclei training diameter={diameter}'):
            pred, _, _ = model.eval(normalise(img), channels=[0, 0], diameter=diameter, normalize=False)
            score = mean_ap(gt, pred)
            values.append(score)
            records.append(dict(stem=row['stem'], split='training', diameter=diameter, map=score, n_gt=int(gt.max()), n_pred=int(pred.max())))
        means[diameter] = float(np.mean(values))
    shortlist = sorted(DIAMETERS, key=lambda d: (-means[d], d))[:3]
    validation_means = {}
    for diameter in shortlist:
        values = []
        for row in tqdm(validation, desc=f'nuclei validation diameter={diameter}'):
            img = read_zarr_image(row['ome_zarr'])
            gt = read_zarr_labels(row['ome_zarr'], 'ground_truth')
            pred, _, _ = model.eval(normalise(img), channels=[0, 0], diameter=diameter, normalize=False)
            score = mean_ap(gt, pred)
            values.append(score)
            records.append(dict(stem=row['stem'], split='validation', diameter=diameter, map=score, n_gt=int(gt.max()), n_pred=int(pred.max())))
        validation_means[diameter] = float(np.mean(values))
    selected = min(shortlist, key=lambda d: (-validation_means[d], -means[d], d))
    with (results / 'legacy_tuning_scores.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stem', 'split', 'diameter', 'map', 'n_gt', 'n_pred'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    record = dict(diameter=selected, candidates=DIAMETERS, shortlist=shortlist,
                  training_mean_map=means, validation_mean_map=validation_means,
                  n_training=len(training), n_validation=len(validation),
                  objective='mean image AP at IoU 0.50:0.05:0.95', empty_case='AP=1 when GT and prediction are both empty',
                  selection='top 3 training candidates, best validation mAP; ties use training mAP then diameter')
    (results / 'legacy_selection.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--diameter', type=float)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    with args.pairs.open() as handle:
        rows = list(csv.DictReader(handle))
    model = models.CellposeModel(gpu=True, model_type='nuclei')
    sha256 = model_digest(model)
    diameter = choose_diameter(model, rows, args.results) if args.tune else args.diameter
    if diameter is None:
        raise ValueError('Provide --tune with a split manifest, or --diameter')
    pairs = [r for r in rows if r.get('split', 'test') == 'test']
    warmup = next((r for r in rows if r.get('split') == 'training'), pairs[0])
    model.eval(normalise(read_zarr_image(warmup['ome_zarr'])), channels=[0, 0], diameter=diameter, normalize=False)
    synchronize(model.device)
    hardware = hardware_record(model.device)
    runtime_rows = []
    for row in tqdm(pairs, desc='Cellpose 3 nuclei test'):
        group = zarr.open_group(row['ome_zarr'], mode='a')
        img = read_zarr_image(row['ome_zarr'])
        synchronize(model.device)
        start = time.perf_counter()
        pred, _, _ = model.eval(normalise(img), channels=[0, 0], diameter=diameter, normalize=False)
        synchronize(model.device)
        runtime_rows.append(dict(stem=row['stem'], method='nuclei',
                                 seconds=time.perf_counter() - start, device=str(model.device)))
        if 'labels/nuclei' in group:
            del group['labels/nuclei']
        transforms = [[dict(type='scale', scale=[p * factor for p in group.attrs['pixel_size_um']])]
                      for factor in [1, 2, 4]]
        write_labels(
            pred.astype(np.int32), group, name='nuclei', scaler=None, scale_factors=(2, 4),
            axes=[dict(name=a, type='space', unit='micrometer') for a in 'yx'],
            coordinate_transformations=transforms, label_metadata={'source': {'image': '../../'}},
        )
        group['labels/nuclei'].attrs['parameters'] = dict(
            model='nuclei', diameter=diameter, channels=[0, 0], sha256=sha256,
            normalization_percentiles=[1, 99.8],
        )
    with (args.results / 'legacy_runtime.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stem', 'method', 'seconds', 'device'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(runtime_rows)
    record = dict(hardware=hardware, model='nuclei', sha256=sha256, diameter=diameter,
                  timing='indicative; untimed warm-up; synchronize before/after; preprocessing included; model loading, process startup and I/O excluded',
                  process='separate conda environment',
                  packages={p: md.version(p) for p in ['cellpose', 'numpy', 'torch', 'ome-zarr', 'zarr']})
    (args.results / 'legacy_run_record.json').write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
