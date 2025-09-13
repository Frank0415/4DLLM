"""File-based lock manager for preventing concurrent background tasks."""
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lock file path
LOCK_FILE_PATH = Path("/tmp/4dllm_analysis.lock")

class LockManager:
    """Manages file-based locking for background analysis tasks."""
    
    @staticmethod
    def acquire_lock(job_info: Dict[str, Any]) -> bool:
        """
        Attempt to acquire the lock file.
        
        Args:
            job_info: Dictionary containing job information (job_id, timestamp, etc.)
            
        Returns:
            bool: True if lock acquired, False if already locked
        """
        try:
            # Add timestamp to job info
            job_info["timestamp"] = time.time()
            
            # Try to create the lock file atomically
            # O_CREAT | O_EXCL ensures atomic creation - fails if file already exists
            fd = os.open(LOCK_FILE_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            
            # Write job info to the lock file
            job_info_str = json.dumps(job_info)
            os.write(fd, job_info_str.encode('utf-8'))
            os.close(fd)
            
            logger.info(f"Lock acquired for job: {job_info.get('job_id', 'unknown')}")
            return True
            
        except FileExistsError:
            # Lock file already exists
            logger.warning("Lock file already exists")
            return False
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            return False
    
    @staticmethod
    def release_lock() -> bool:
        """
        Release the lock by deleting the lock file.
        
        Returns:
            bool: True if lock released, False if error occurred
        """
        try:
            if LOCK_FILE_PATH.exists():
                LOCK_FILE_PATH.unlink()
                logger.info("Lock released")
                return True
            else:
                logger.warning("Lock file does not exist when trying to release")
                return True
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            return False
    
    @staticmethod
    def is_locked() -> bool:
        """
        Check if the system is currently locked.
        
        Returns:
            bool: True if locked, False otherwise
        """
        return LOCK_FILE_PATH.exists()
    
    @staticmethod
    def get_lock_info() -> Optional[Dict[str, Any]]:
        """
        Get information about the current lock holder.
        
        Returns:
            dict: Lock information or None if not locked
        """
        if not LOCK_FILE_PATH.exists():
            return None
            
        try:
            with open(LOCK_FILE_PATH, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error reading lock file: {e}")
            return None
    
    @staticmethod
    def check_and_clean_stale_lock() -> bool:
        """
        Check if the lock is stale (process no longer running) and clean it if so.
        
        Returns:
            bool: True if lock was stale and cleaned, False otherwise
        """
        lock_info = LockManager.get_lock_info()
        if not lock_info:
            return False
            
        # For now, we'll just check the timestamp - in a real implementation
        # we would check if the process is still running
        timestamp = lock_info.get("timestamp", 0)
        current_time = time.time()
        
        # If the lock is older than 1 hour, consider it stale
        if current_time - timestamp > 3600:  # 1 hour
            logger.warning(f"Stale lock detected (older than 1 hour), cleaning up")
            LockManager.release_lock()
            return True
        
        return False