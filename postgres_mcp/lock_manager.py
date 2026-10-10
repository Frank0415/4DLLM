import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from .workflows import data_root


class LockManager:
    @staticmethod
    def path():
        return data_root() / 'analysis.lock'

    @staticmethod
    def acquire_lock(job_info):
        path = LockManager.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, 'w') as source:
            json.dump({**job_info, 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat()}, source)
        return True

    @staticmethod
    def release_lock():
        info = LockManager.get_lock_info()
        if info and info.get('pid') == os.getpid():
            LockManager.path().unlink(missing_ok=True)
        return True

    @staticmethod
    def is_locked():
        return LockManager.path().exists()

    @staticmethod
    def get_lock_info():
        try:
            return json.loads(LockManager.path().read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @staticmethod
    def check_and_clean_stale_lock():
        info = LockManager.get_lock_info()
        if not info:
            return False
        try:
            os.kill(info['pid'], 0)
        except ProcessLookupError:
            LockManager.path().unlink(missing_ok=True)
            return True
        except PermissionError:
            pass
        return False

    @staticmethod
    @contextmanager
    def hold(job_info):
        LockManager.check_and_clean_stale_lock()
        if not LockManager.acquire_lock(job_info):
            raise RuntimeError('Another clustering task is running. Try again when it completes')
        try:
            yield
        finally:
            LockManager.release_lock()
