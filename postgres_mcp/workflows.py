import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from .config.db_config import PROJECT_ROOT
from .import_4dstem.import_mib import inspect_mib, process_one_mib


def data_root():
    return Path(os.environ.get('FOURDLLM_DATA_DIR', PROJECT_ROOT / 'workspace')).expanduser().resolve()


async def initialize_schema(pool):
    async with pool.connection() as connection:
        async with connection.transaction():
            await connection.execute('SELECT pg_advisory_xact_lock(4140415)')
            await connection.execute((PROJECT_ROOT / 'SQL/schema.sql').read_text(encoding='utf-8'))


async def ingest_scan(sql_driver, source_mib_path, scan_size=None, scan_name=None, progress=None):
    source = Path(source_mib_path).expanduser().resolve(strict=True)
    name = (scan_name or source.stem).strip()
    if not name or len(name) > 255:
        raise ValueError('Scan name must contain 1–255 characters')
    info = await asyncio.to_thread(inspect_mib, source, scan_size)
    if info['scan_shape'] is None:
        raise ValueError('Specify scan rows and columns before importing')
    existing = await sql_driver.execute_query('SELECT id FROM scans WHERE scan_name = %s', [name])
    if existing:
        raise ValueError(f"Scan '{name}' already exists. Choose another scan name")
    root = data_root() / 'scans'
    root.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    staging = root / f'.{identifier}.partial'
    final = root / identifier
    staging.mkdir()
    published = False

    def prepare():
        raw = staging / 'Raw'
        raw.mkdir()
        managed = raw / source.name
        digest = hashlib.sha256()
        copied = 0
        with source.open('rb') as src, managed.open('xb') as dst:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                if progress:
                    progress('copy', 0.3 * copied / info['size_bytes'], 'Copying raw data')
        if info['hdr_path']:
            shutil.copy2(info['hdr_path'], managed.with_suffix('.hdr'))
        def converting(stage, fraction, message):
            if progress:
                progress(stage, 0.3 + 0.5 * fraction, message)
        folder = Path(process_one_mib(str(managed), staging / 'Data', info['scan_shape'], converting))
        return digest.hexdigest(), folder.relative_to(staging)

    try:
        digest, relative_folder = await asyncio.to_thread(prepare)
        if progress:
            progress('catalog', 0.8, 'Saving scan catalog')
        async with sql_driver.transaction() as tx:
            duplicate = await tx.execute_query('SELECT scan_name FROM scans WHERE source_sha256 = %s', [digest])
            if duplicate:
                raise ValueError(f"This data is already imported as '{duplicate[0].cells['scan_name']}'")
            staging.rename(final)
            folder = final / relative_folder
            result = await tx.execute_query(
                'INSERT INTO scans (scan_name, folder_path, source_mib_path, source_sha256, metadata) '
                'VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING id',
                [name, str(folder), str(final / 'Raw' / source.name), digest, json.dumps(info)],
            )
            scan_id = result[0].cells['id']
            rows, cols = info['scan_shape']
            for row in range(1, rows + 1):
                if progress:
                    progress('catalog', 0.8 + 0.19 * row / rows, f'Saving row {row}/{rows}')
                mat = await tx.execute_query(
                    'INSERT INTO raw_mat_files (scan_id, row_index, file_path) VALUES (%s, %s, %s) RETURNING id',
                    [scan_id, row, str(folder / f'{row}.mat')],
                )
                await tx.copy_rows('diffraction_patterns', ['source_mat_id', 'col_index'],
                                   ((mat[0].cells['id'], col) for col in range(1, cols + 1)))
        published = True
        return {'scan_id': scan_id, 'scan_name': name, 'total_patterns': rows * cols, 'scan_shape': [rows, cols]}
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(final, ignore_errors=True)
