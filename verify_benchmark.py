from pathlib import Path
import tempfile

import nbformat
import numpy as np
import pandas as pd
import zarr

from benchmark_utils import level_zero, read_zarr_image, read_zarr_labels


def main():
    notebook = nbformat.read('run_benchmark.ipynb', as_version=4)
    nbformat.validate(notebook)
    assert all(output.output_type != 'error' for cell in notebook.cells if cell.cell_type == 'code'
               for output in cell.outputs)
    counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == 'code']
    assert counts == list(range(1, len(counts) + 1)), (
        f'Saved outputs are not from one top-to-bottom run (execution counts {counts}); '
        'use Restart Kernel and Run All Cells, then save')
    for version in [2, 3]:
        with tempfile.TemporaryDirectory() as tmp:
            group = zarr.open_group(tmp, mode='w', zarr_format=version)
            image = np.arange(12).reshape(3, 4)
            group.create_array('image', data=image)
            metadata = {'multiscales': [{'datasets': [{'path': 'image'}]}]}
            group.attrs.update(metadata if version == 2 else {'ome': {'version': '0.5', **metadata}})
            np.testing.assert_array_equal(level_zero(group), image)
    root = Path('bbbc039_demo/results')
    pairs = pd.read_csv(root / 'all_pairs.csv')
    assert pairs.groupby('split').size().to_dict() == {'test': 50, 'training': 100, 'validation': 50}
    assert pairs.stem.nunique() == 200
    test = pairs.loc[pairs.split == 'test']
    test_stems = set(test.stem)
    import json
    watershed_selection = json.loads((root / 'watershed_selection.json').read_text())
    legacy_selection = json.loads((root / 'legacy_selection.json').read_text())
    for file, candidate_column, n_candidates in [('watershed_tuning_scores.csv', 'candidate', 27),
                                                 ('legacy_tuning_scores.csv', 'diameter', 5)]:
        scores = pd.read_csv(root / file)
        assert np.isfinite(scores['map']).all(), 'Non-finite tuning score'
        assert not set(scores.stem) & test_stems
        assert scores.groupby('split').stem.nunique().to_dict() == {'training': 100, 'validation': 50}
        assert scores.loc[scores.split == 'training', candidate_column].nunique() == n_candidates
        assert scores.loc[scores.split == 'validation', candidate_column].nunique() == 3
        assert len(scores) == n_candidates * 100 + 150
    candidates = watershed_selection['shortlist']
    selected = min(candidates, key=lambda i: (-watershed_selection['validation_mean_map'][str(i)],
                                             -watershed_selection['training_mean_map'][i], i))
    assert selected == watershed_selection['selected_candidate']
    selected_diameter = min(legacy_selection['shortlist'], key=lambda d: (
        -legacy_selection['validation_mean_map'][str(d)], -legacy_selection['training_mean_map'][str(d)], d))
    assert selected_diameter == legacy_selection['diameter']
    for row in test.itertuples(index=False):
        group = zarr.open_group(row.ome_zarr, mode='r')
        image = read_zarr_image(row.ome_zarr)
        image_meta = group.attrs['ome']['multiscales'][0]
        for method in ['ground_truth', 'watershed', 'cpsam_v2', 'nuclei']:
            labels = read_zarr_labels(row.ome_zarr, method)
            assert labels.shape == image.shape and np.issubdtype(labels.dtype, np.integer)
            label_group = group[f'labels/{method}']
            metadata = label_group.attrs['ome']['multiscales'][0]
            assert metadata['axes'] == image_meta['axes'] and metadata['datasets'] == image_meta['datasets']
        assert group['labels/watershed'].attrs['parameters'] == watershed_selection['parameters']
        assert group['labels/nuclei'].attrs['parameters']['diameter'] == selected_diameter
    legacy_scores = pd.read_csv(root / 'legacy_tuning_scores.csv')
    empty = (legacy_scores.n_gt == 0) & (legacy_scores.n_pred == 0)
    assert empty.any() and legacy_scores.loc[empty, 'map'].eq(1).all()
    metrics = pd.read_csv(root / 'image_level_metrics.csv')
    ap = pd.read_csv(root / 'ap_by_image_threshold.csv')
    assert len(metrics) == 150 and metrics.groupby('method').stem.nunique().eq(50).all()
    assert len(ap) == 1500 and ap.groupby(['stem', 'method']).iou.nunique().eq(10).all()
    np.testing.assert_allclose(ap.ap, ap.tp / (ap.tp + ap.fp + ap.fn), atol=1e-7)
    summary = pd.read_csv(root / 'summary_bootstrap.csv')
    for method in metrics.method.unique():
        mean = summary.loc[(summary.method == method) & (summary.metric == 'map'), 'mean'].iloc[0]
        np.testing.assert_allclose(mean, metrics.loc[metrics.method == method, 'map'].mean())
    hardware = json.loads((root / 'hardware.json').read_text())
    runtimes = pd.read_csv(root / 'runtime.csv')
    assert len(runtimes) == 150
    for method in metrics.method.unique():
        assert hardware[method]['cpu'] and hardware[method]['device']
        assert runtimes.loc[runtimes.method == method, 'device'].eq(hardware[method]['device']).all()
    audit = pd.read_csv(root / 'mask_decoding_audit.csv')
    assert len(audit) == 200 and audit.same_partition.all()
    features = pd.read_parquet(root / 'nucleus_features.parquet')
    assert 'intensity_mean' in features and 'mean_intensity' not in features
    assert (root / 'results.md').read_text().strip()
    print('Passed: single top-to-bottom notebook run; both NGFF layouts; train/validation-only selection; 50 complete test stores; 1,500 AP scores; hardware/timing records; decoder audit; current feature names.')


if __name__ == '__main__':
    main()
