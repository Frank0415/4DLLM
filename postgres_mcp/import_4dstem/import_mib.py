from pathlib import Path

import numpy as np
import scipy.io


def inspect_mib(file_path, scan=None):
    path = Path(file_path)
    if path.suffix.lower() != '.mib':
        raise ValueError('Select a .mib file')
    with path.open('rb') as source:
        prefix = source.read(128).split(b',')
        if len(prefix) < 4:
            raise ValueError('Incomplete MIB header')
        offset = int(prefix[2])
        if not 0 < offset <= 65536:
            raise ValueError('Invalid MIB header length')
        source.seek(0)
        fields = source.read(offset).split(b'\0', 1)[0].decode('ascii').split(',')
    if len(fields) < 8:
        raise ValueError('Incomplete MIB header')
    formats = {'U08': ('u1', 6), 'U16': ('>u2', 12), 'U32': ('>u4', 24)}
    if fields[6] not in formats:
        raise ValueError(f'Unsupported MIB format: {fields[6]}. Supported: U08, U16, U32')
    layouts = {'1x1': 256, '2x2': 512, '2x2G': 514}
    layout = fields[7].strip()
    if layout not in layouts:
        raise ValueError(f'Unsupported detector layout: {layout}')
    width = layouts[layout]
    dtype, bit_depth = formats[fields[6]]
    dtype = np.dtype(dtype)
    frame_bytes = offset + width * width * dtype.itemsize
    size = path.stat().st_size
    frames, remainder = divmod(size, frame_bytes)
    if remainder or frames == 0:
        raise ValueError('MIB size does not contain complete detector frames')
    if offset % dtype.itemsize:
        raise ValueError('MIB header is not aligned with pixel size')
    detected_scan = None
    hdr = path.with_suffix('.hdr')
    if not hdr.exists():
        hdr = path.with_suffix('.HDR')
    if hdr.exists():
        values = {}
        for line in hdr.read_text(encoding='utf-8-sig').splitlines():
            parts = line.split('\t', 1)
            if len(parts) == 2:
                values[parts[0].rstrip(':').strip()] = parts[1].strip()
        if 'ScanX' in values and 'ScanY' in values:
            detected_scan = (int(values['ScanY']), int(values['ScanX']))
    if detected_scan is None and len(fields) > 19:
        try:
            candidate = (int(fields[19]), int(fields[18]))
            if min(candidate) > 0 and candidate[0] * candidate[1] == frames:
                detected_scan = candidate
        except ValueError:
            pass
    resolved = tuple(scan) if scan is not None else detected_scan
    if resolved is None and frames == 256 * 256:
        resolved = (256, 256)
    if resolved is not None and (len(resolved) != 2 or min(resolved) < 1 or resolved[0] * resolved[1] != frames):
        raise ValueError(f'Scan rows × columns must equal {frames} frames')
    return {'path': str(path.resolve()), 'filename': path.name, 'size_bytes': size,
            'frame_count': frames, 'header_bytes': offset, 'dtype': dtype.str,
            'bit_depth': bit_depth, 'detector_shape': [width, width],
            'scan_shape': list(resolved) if resolved else None,
            'hdr_path': str(hdr.resolve()) if hdr.exists() else None}


def load_mib(file_path, mem='MEMMAP', binfactor=1, reshape=True, scan=None):
    if binfactor != 1:
        raise ValueError('MIB binning is not implemented')
    info = inspect_mib(file_path, scan)
    dtype = np.dtype(info['dtype'])
    height, width = info['detector_shape']
    header_pixels = info['header_bytes'] // dtype.itemsize
    raw = np.memmap(file_path, dtype=dtype, mode='r')
    data = raw.reshape(info['frame_count'], header_pixels + height * width)[:, header_pixels:]
    data = data.reshape(info['frame_count'], height, width)
    if reshape:
        if info['scan_shape'] is None:
            raise ValueError(f"Scan dimensions are missing. Specify rows and columns for {info['frame_count']} frames")
        data = data.reshape(*info['scan_shape'], height, width)
    else:
        data = data[:, None, :, :]
    return np.array(data) if mem == 'RAM' else data


def process_one_mib(mib_path, out_root, scan_size=None, progress=None):
    data = load_mib(mib_path, scan=scan_size)
    folder = Path(out_root) / Path(mib_path).stem
    folder.mkdir(parents=True, exist_ok=True)
    preview = np.empty(data.shape[:2], dtype=np.float64)
    for row in range(data.shape[0]):
        if progress:
            progress('convert', row / data.shape[0], f'Converting row {row + 1}/{data.shape[0]}')
        scipy.io.savemat(folder / f'{row + 1}.mat', {'data': data[row]})
        preview[row] = np.sum(data[row], axis=(1, 2), dtype=np.float64)
    np.save(folder / "scan_preview.npy", preview)
    return str(folder)
