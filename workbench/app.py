import asyncio
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import scipy.io
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from postgres_mcp.config import ConfigManager
from postgres_mcp.config.db_config import PROJECT_ROOT
from postgres_mcp.import_4dstem.import_mib import inspect_mib, load_mib
from postgres_mcp.sql import DbConnPool, SqlDriver
from postgres_mcp.workflows import data_root, initialize_schema
from .jobs import JobManager, read_job

pool = DbConnPool()
jobs = None
connection_lock = asyncio.Lock()
settings_path = data_root() / 'settings.json'


def save_json(path, value, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def settings():
    if settings_path.exists():
        return json.loads(settings_path.read_text(encoding='utf-8'))
    return {'import_roots': [str(Path.home() / 'Documents')]}


def allowed_path(value):
    path = Path(value).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in settings()['import_roots']]
    if not any(path.is_relative_to(root) for root in roots):
        raise HTTPException(403, 'Select a path inside a configured import directory')
    if not path.exists():
        raise HTTPException(404, 'File or directory does not exist')
    return path


async def database():
    async with connection_lock:
        if not pool.is_valid:
            try:
                await asyncio.wait_for(pool.pool_connect(ConfigManager().get_database_url()), timeout=8)
                await initialize_schema(pool.pool)
            except Exception as error:
                await pool.close()
                raise HTTPException(503, 'Database is unavailable. Configure PostgreSQL in Settings') from error
    return SqlDriver(conn=pool)


@asynccontextmanager
async def lifespan(app):
    global jobs
    jobs = JobManager()
    await jobs.start()
    yield
    await jobs.close()
    await pool.close()


app = FastAPI(title='4DLLM Workbench', lifespan=lifespan)


@app.middleware('http')
async def local_requests(request: Request, call_next):
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('localhost', '127.0.0.1'):
        return Response('Local access only', status_code=403)
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
        return Response('Same-origin requests required', status_code=403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def invalid_input(request, error):
    from fastapi.responses import JSONResponse
    return JSONResponse({'detail': str(error)}, status_code=400)


@app.exception_handler(FileNotFoundError)
async def missing_file(request, error):
    from fastapi.responses import JSONResponse
    return JSONResponse({'detail': str(error)}, status_code=404)


@app.get('/api/status')
async def status():
    error = None
    try:
        await database()
    except HTTPException as problem:
        error = problem.detail
    config = ConfigManager().get_api_keys() or {}
    return {'database_ready': error is None, 'database_error': error,
            'llm_ready': bool(config.get('api_keys')) and not any('your-api-key' in key for key in config['api_keys']),
            'storage_directory': str(data_root())}


@app.get('/api/settings')
async def get_settings():
    config = ConfigManager()
    try:
        config.load_config()
        db = {key: value for key, value in config.config.items() if key != 'password'}
        db['password_configured'] = bool(config.config.get('password'))
    except FileNotFoundError:
        db = {'host': 'localhost', 'port': 5432, 'user': 'postgres', 'dbname': '4dllm', 'password_configured': False}
    llm = config.get_api_keys() or {}
    return {**settings(), 'database': db, 'llm': {'base_url': llm.get('base_url', 'https://generativelanguage.googleapis.com'),
            'model': llm.get('model', 'gemini-2.5-pro'), 'keys_configured': bool(llm.get('api_keys'))}}


class DatabaseSettings(BaseModel):
    host: str = 'localhost'
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = 'postgres'
    dbname: str = '4dllm'
    password: str | None = None


class LLMSettings(BaseModel):
    base_url: str = 'https://generativelanguage.googleapis.com'
    model: str = 'gemini-2.5-pro'
    api_keys: list[str] | None = None


class SettingsUpdate(BaseModel):
    import_roots: list[str]
    database: DatabaseSettings
    llm: LLMSettings


@app.put('/api/settings')
async def save_settings(value: SettingsUpdate):
    if not value.import_roots:
        raise ValueError('Configure at least one import directory')
    roots = [str(Path(root).expanduser().resolve(strict=True)) for root in value.import_roots]
    if not all(Path(root).is_dir() for root in roots):
        raise ValueError('Import roots must be directories')
    if jobs and any(job['status'] in ('running', 'queued') for job in jobs.list()):
        raise HTTPException(409, 'Wait for pending tasks before changing settings')
    manager = ConfigManager()
    try:
        manager.load_config()
    except FileNotFoundError:
        pass
    db = value.database.model_dump()
    if db['password'] is None:
        db['password'] = manager.config.get('password', '')
    llm = value.llm.model_dump()
    if llm['api_keys'] is None:
        llm['api_keys'] = (manager.get_api_keys() or {}).get('api_keys', [])
    save_json(manager.primary_config_path, db, private=True)
    save_json(PROJECT_ROOT / 'config/api_keys.json', llm, private=True)
    save_json(settings_path, {'import_roots': roots})
    async with connection_lock:
        await pool.close()
    return {'saved': True}


@app.get('/api/files')
async def files(path: str | None = None):
    roots = settings()['import_roots']
    directory = allowed_path(path or roots[0])
    if not directory.is_dir():
        raise ValueError('Select a directory')
    entries = []
    for entry in directory.iterdir():
        if entry.name.startswith('.'):
            continue
        try:
            if entry.is_dir() or entry.suffix.lower() == '.mib':
                resolved = entry.resolve()
                if not any(resolved.is_relative_to(Path(root).resolve()) for root in roots):
                    continue
                entries.append({'name': entry.name, 'path': str(entry), 'directory': entry.is_dir(),
                                'size_bytes': 0 if entry.is_dir() else entry.stat().st_size})
        except OSError:
            continue
    return {'path': str(directory), 'roots': roots, 'entries': sorted(entries, key=lambda e: (not e['directory'], e['name'].lower()))}


class InspectRequest(BaseModel):
    path: str
    scan_shape: tuple[int, int] | None = None


@app.post('/api/inspect')
async def inspect(value: InspectRequest):
    return await asyncio.to_thread(inspect_mib, allowed_path(value.path), value.scan_shape)


def png(array):
    image = np.asarray(array, dtype=np.float32)
    image = np.log1p(np.maximum(image, 0))
    lo, hi = np.percentile(image, [1, 99.5])
    image = np.clip((image - lo) / max(float(hi - lo), 1e-8), 0, 1)
    buffer = io.BytesIO()
    Image.fromarray((image * 255).astype(np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()


@app.get('/api/source-preview')
async def source_preview(path: str):
    def render():
        data = load_mib(str(allowed_path(path)), reshape=False)
        return png(data[0, 0])
    return Response(await asyncio.to_thread(render), media_type='image/png')


class ImportRequest(InspectRequest):
    scan_name: str | None = None


@app.post('/api/imports', status_code=202)
async def import_scan(value: ImportRequest):
    await database()
    path = allowed_path(value.path)
    info = await asyncio.to_thread(inspect_mib, path, value.scan_shape)
    if info['scan_shape'] is None:
        raise ValueError('Specify scan rows and columns')
    return await jobs.submit('import', {'path': str(path), 'scan_shape': info['scan_shape'], 'scan_name': value.scan_name})


class FolderImport(BaseModel):
    path: str
    scan_shape: tuple[int, int] | None = None


@app.post('/api/import-folders', status_code=202)
async def import_folder(value: FolderImport):
    await database()
    folder = allowed_path(value.path)
    if not folder.is_dir():
        raise ValueError('Select a directory')
    sources = sorted(path for path in folder.iterdir() if path.suffix.lower() == '.mib' and path.is_file())
    if not sources:
        raise ValueError('No .mib files in this directory')
    prepared = []
    for path in sources:
        info = await asyncio.to_thread(inspect_mib, allowed_path(str(path)), value.scan_shape)
        if info['scan_shape'] is None:
            raise ValueError(f'Specify scan dimensions for {path.name}')
        prepared.append((path, info['scan_shape']))
    return {'jobs': [await jobs.submit('import', {'path': str(path), 'scan_shape': shape}) for path, shape in prepared]}


@app.get('/api/scans')
async def scans():
    driver = await database()
    rows = await driver.execute_query(
        'SELECT s.id, s.scan_name, s.created_at, s.metadata, '
        '(SELECT COUNT(*) FROM diffraction_patterns dp JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id WHERE rmf.scan_id = s.id) AS pattern_count, '
        '(SELECT COUNT(*) FROM clustering_runs cr WHERE cr.scan_id = s.id) AS run_count '
        'FROM scans s ORDER BY s.created_at DESC')
    return [row.cells for row in rows or []]


async def scan_record(identifier):
    driver = await database()
    rows = await driver.execute_query('SELECT * FROM scans WHERE id = %s', [identifier])
    if not rows:
        raise HTTPException(404, 'Scan not found')
    return rows[0].cells


@app.get('/api/scans/{identifier}')
async def scan_details(identifier: int):
    scan = await scan_record(identifier)
    driver = await database()
    rows = await driver.execute_query(
        'SELECT row_index, file_path FROM raw_mat_files WHERE scan_id = %s ORDER BY row_index', [identifier])
    entries = [row.cells for row in rows or []]
    if not entries:
        raise ValueError('Scan contains no rows')
    shape = scan['metadata'].get('scan_shape')
    if not shape:
        shape = [max(entry['row_index'] for entry in entries),
                 await asyncio.to_thread(lambda: len(scipy.io.loadmat(entries[0]['file_path'])['data']))]
    scan['scan_shape'] = shape
    runs = await driver.execute_query('SELECT * FROM clustering_runs WHERE scan_id = %s ORDER BY id DESC', [identifier])
    scan['runs'] = [row.cells for row in runs or []]
    return scan


@app.get('/api/scans/{identifier}/overview')
async def overview(identifier: int):
    scan = await scan_record(identifier)
    def render():
        path = Path(scan['folder_path']) / 'scan_preview.npy'
        if path.exists():
            return png(np.load(path))
        files = sorted(Path(scan['folder_path']).glob('*.mat'), key=lambda p: int(p.stem))
        return png(np.stack([np.sum(scipy.io.loadmat(path)['data'], axis=(1, 2), dtype=np.float64) for path in files]))
    return Response(await asyncio.to_thread(render), media_type='image/png')


@app.get('/api/scans/{identifier}/pattern')
async def pattern(identifier: int, row: int = 1, col: int = 1):
    driver = await database()
    rows = await driver.execute_query('SELECT file_path FROM raw_mat_files WHERE scan_id = %s AND row_index = %s', [identifier, row])
    if not rows:
        raise HTTPException(404, 'Scan row not found')
    def render():
        data = scipy.io.loadmat(rows[0].cells['file_path'])['data']
        if not 1 <= col <= len(data):
            raise ValueError('Column is outside the scan')
        return png(data[col - 1])
    return Response(await asyncio.to_thread(render), media_type='image/png')


class ClusterRequest(BaseModel):
    k: int = Field(default=16, ge=1, le=256)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    device: str | None = None


@app.post('/api/scans/{identifier}/cluster', status_code=202)
async def cluster(identifier: int, value: ClusterRequest):
    await scan_record(identifier)
    if value.device not in (None, 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu, or cuda')
    return await jobs.submit('cluster', {'scan_id': identifier, **value.model_dump()})


class LLMRequest(BaseModel):
    cluster_id: int | None = None
    samples: int = Field(default=5, ge=1, le=16)
    concurrency: int = Field(default=3, ge=1, le=8)


@app.post('/api/scans/{identifier}/llm', status_code=202)
async def llm(identifier: int, value: LLMRequest):
    await scan_record(identifier)
    driver = await database()
    if value.cluster_id is not None:
        runs = await driver.execute_query(
            'SELECT cr.id FROM clustering_runs cr JOIN identified_clusters ic ON ic.run_id = cr.id '
            'WHERE cr.scan_id = %s AND ic.id = %s', [identifier, value.cluster_id])
    else:
        runs = await driver.execute_query('SELECT id FROM clustering_runs WHERE scan_id = %s ORDER BY id DESC LIMIT 1', [identifier])
    if not runs:
        raise ValueError('Run clustering before LLM analysis')
    return await jobs.submit('llm', {'scan_id': identifier, 'run_id': runs[0].cells['id'], **value.model_dump()})


async def run_record(identifier):
    driver = await database()
    rows = await driver.execute_query('SELECT * FROM clustering_runs WHERE id = %s', [identifier])
    if not rows:
        raise HTTPException(404, 'Analysis run not found')
    return rows[0].cells


@app.get('/api/runs/{identifier}')
async def run_details(identifier: int):
    record = await run_record(identifier)
    driver = await database()
    rows = await driver.execute_query(
        'SELECT ic.id, ic.cluster_index, ic.pattern_count, ca.category AS manual_category, ca.note AS manual_note, '
        'la.llm_assigned_class, la.llm_detailed_features FROM identified_clusters ic '
        'LEFT JOIN cluster_annotations ca ON ca.cluster_id = ic.id LEFT JOIN llm_analyses la ON la.cluster_id = ic.id '
        'WHERE ic.run_id = %s ORDER BY ic.cluster_index', [identifier])
    record['clusters'] = [row.cells for row in rows or []]
    record['files'] = sorted(str(path.relative_to(record['results_path'])) for path in Path(record['results_path']).rglob('*')
                             if path.is_file() and path.suffix in ('.png', '.npz', '.json'))
    return record


@app.get('/api/runs/{identifier}/grid')
async def grid(identifier: int):
    record = await run_record(identifier)
    def load():
        candidates = list(Path(record['results_path']).glob('*.npz'))
        if not candidates:
            raise FileNotFoundError('Cluster arrays not found')
        with np.load(candidates[0]) as data:
            labels, xs, ys = data['labels'], data['xs'], data['ys']
        shape = [int(xs.max()), int(ys.max())]
        array = np.full(shape, -1, dtype=np.int32)
        array[xs - 1, ys - 1] = labels
        return {'shape': shape, 'labels': array.ravel().tolist()}
    return await asyncio.to_thread(load)


@app.get('/api/runs/{identifier}/files/{name:path}')
async def artifact(identifier: int, name: str):
    record = await run_record(identifier)
    folder = Path(record['results_path']).resolve()
    path = (folder / name).resolve()
    if not path.is_relative_to(folder) or path.suffix not in ('.png', '.npz', '.json') or not path.is_file():
        raise HTTPException(404, 'Result file not found')
    return FileResponse(path, filename=path.name if path.suffix != '.png' else None)


class Annotation(BaseModel):
    category: str | None = None
    note: str = ''


@app.put('/api/clusters/{identifier}/annotation')
async def annotate(identifier: int, value: Annotation):
    driver = await database()
    if value.category is None:
        await driver.execute_query('DELETE FROM cluster_annotations WHERE cluster_id = %s', [identifier])
    else:
        if value.category not in ('vacuum', 'crystalline', 'amorphous', 'mixed'):
            raise ValueError('Choose a supported material category')
        await driver.execute_query(
            'INSERT INTO cluster_annotations (cluster_id, category, note) VALUES (%s, %s, %s) '
            'ON CONFLICT (cluster_id) DO UPDATE SET category = EXCLUDED.category, note = EXCLUDED.note, updated_at = CURRENT_TIMESTAMP',
            [identifier, value.category, value.note])
    return {'saved': True}


@app.get('/api/jobs')
async def job_list():
    return jobs.list()


@app.get('/api/jobs/{identifier}')
async def job_detail(identifier: str):
    return read_job(jobs.path(identifier))


@app.post('/api/jobs/{identifier}/cancel')
async def cancel_job(identifier: str):
    return jobs.cancel(identifier)


@app.post('/api/jobs/{identifier}/retry', status_code=202)
async def retry_job(identifier: str):
    job = read_job(jobs.path(identifier))
    if job['status'] not in ('failed', 'cancelled'):
        raise HTTPException(409, 'Only failed or cancelled tasks can be retried')
    await database()
    return await jobs.submit(job['kind'], job['parameters'])


@app.get('/api/jobs/{identifier}/report')
async def report(identifier: str):
    return FileResponse(jobs.path(identifier), media_type='application/json', filename=f'4dllm-{identifier}.json')


@app.get('/api/jobs/{identifier}/log')
async def job_log(identifier: str):
    path = jobs.path(identifier).with_suffix('.log')
    if not path.exists():
        return {'text': ''}
    def tail():
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size - 16000))
            return source.read().decode('utf-8', errors='replace')
    return {'text': await asyncio.to_thread(tail)}


frontend = PROJECT_ROOT / 'frontend/dist'
if frontend.exists():
    app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')
else:
    @app.get('/')
    async def build_notice():
        return Response('Build the UI with ./run.sh, or cd frontend && npm ci && npm run build', media_type='text/plain')
