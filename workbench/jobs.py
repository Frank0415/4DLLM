import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from postgres_mcp.config.db_config import PROJECT_ROOT
from postgres_mcp.workflows import data_root

TERMINAL = {'succeeded', 'failed', 'cancelled'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_job(path, job):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_job(path):
    job = json.loads(path.read_text(encoding='utf-8'))
    job['cancel_requested'] = path.with_suffix('.cancel').exists()
    return job


class JobCancelled(Exception):
    pass


class JobProgress:
    def __init__(self, path):
        self.path = path
        self.last_update = 0.0

    def __call__(self, stage, fraction, message):
        if self.path.with_suffix('.cancel').exists():
            raise JobCancelled()
        current = time.monotonic()
        if current - self.last_update < 0.3:
            return
        job = read_job(self.path)
        job.update(stage=stage, progress=round(max(0, min(1, fraction)), 4), message=message, updated_at=now())
        write_job(self.path, job)
        self.last_update = current


class JobManager:
    def __init__(self):
        self.directory = data_root() / 'jobs'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.queue = asyncio.Queue()
        self.runner = None
        self.process = None
        self.stopping = False

    def path(self, identifier):
        try:
            identifier = uuid4().hex if identifier is None else UUID(identifier).hex
        except ValueError:
            raise FileNotFoundError('Job not found')
        return self.directory / f'{identifier}.json'

    def list(self):
        return sorted((read_job(path) for path in self.directory.glob('*.json')), key=lambda job: job['created_at'], reverse=True)

    async def start(self):
        for job in reversed(self.list()):
            if job['status'] in ('queued', 'running'):
                await self.queue.put(job['id'])
        self.runner = asyncio.create_task(self._run())

    @staticmethod
    def alive(pid):
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    async def submit(self, kind, parameters):
        identifier = uuid4().hex
        job = {'id': identifier, 'kind': kind, 'parameters': parameters, 'status': 'queued',
               'progress': 0, 'stage': 'queued', 'message': 'Waiting', 'created_at': now(), 'updated_at': now()}
        write_job(self.path(identifier), job)
        await self.queue.put(identifier)
        return job

    def cancel(self, identifier):
        path = self.path(identifier)
        job = read_job(path)
        if job['status'] in TERMINAL:
            return job
        path.with_suffix('.cancel').touch()
        if job['status'] == 'queued':
            job.update(status='cancelled', updated_at=now())
            write_job(path, job)
        return read_job(path)

    async def _run(self):
        while not self.stopping:
            identifier = await self.queue.get()
            path = self.path(identifier)
            try:
                job = read_job(path)
                if job['status'] == 'running':
                    while job['status'] == 'running' and self.alive(job.get('pid')):
                        await asyncio.sleep(1)
                        job = read_job(path)
                    if job['status'] == 'running':
                        job.update(status='failed', error='Worker stopped before the task completed', updated_at=now())
                        write_job(path, job)
                    continue
                if job['status'] != 'queued':
                    continue
                with path.with_suffix('.log').open('ab') as log:
                    self.process = await asyncio.create_subprocess_exec(
                        sys.executable, '-m', 'workbench.worker', str(path), cwd=PROJECT_ROOT,
                        stdout=log, stderr=log)
                    job = read_job(path)
                    if job['status'] not in TERMINAL:
                        job.update(status='running', pid=self.process.pid, stage='starting', updated_at=now())
                        write_job(path, job)
                    code = await self.process.wait()
                job = read_job(path)
                if job['status'] not in TERMINAL:
                    job.update(status='failed', error=f'Worker exited with code {code}', updated_at=now())
                    write_job(path, job)
            except Exception as error:
                job = read_job(path)
                job.update(status='failed', error=str(error), updated_at=now())
                write_job(path, job)
            finally:
                self.process = None
                self.queue.task_done()

    async def close(self):
        self.stopping = True
        if self.process and self.process.returncode is None:
            await self.process.wait()
        if self.runner:
            self.runner.cancel()
            try:
                await self.runner
            except asyncio.CancelledError:
                pass
