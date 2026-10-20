import hashlib
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import zarr


def level_zero(group):
    attrs = group.attrs.asdict()
    path = attrs.get('ome', attrs)['multiscales'][0]['datasets'][0]['path']
    return np.asarray(group[path])


def read_zarr_image(path):
    return level_zero(zarr.open_group(str(path), mode='r'))


def read_zarr_labels(path, name):
    return level_zero(zarr.open_group(str(path), mode='r')[f'labels/{name}'])


def normalise(img):
    low, high = np.percentile(img, [1, 99.8])
    if high <= low:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - float(low)) / float(high - low), 0, 1)


def synchronize(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def model_digest(model):
    if not isinstance(model.pretrained_model, (str, os.PathLike)):
        raise TypeError('Expected one model-weight path for this pinned Cellpose version')
    with Path(model.pretrained_model).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def hardware_record(device):
    name = str(device)
    cpu = platform.processor() or platform.machine()
    memory = None
    if platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        memory = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    accelerator = None
    if name.startswith('cuda'):
        import torch
        accelerator = torch.cuda.get_device_name(device)
    elif name.startswith('mps'):
        accelerator = cpu + ' integrated GPU'
    return dict(device=name, cpu=cpu, accelerator=accelerator, memory_bytes=memory,
                logical_cpus=os.cpu_count(), machine=platform.machine(), os=platform.platform())


def touching_fraction(labels):
    touching = []
    for left, right in [(labels[:-1, :], labels[1:, :]), (labels[:, :-1], labels[:, 1:])]:
        contact = (left > 0) & (right > 0) & (left != right)
        touching.extend([left[contact], right[contact]])
    return len(np.unique(np.concatenate(touching))) / max(int(labels.max()), 1)
