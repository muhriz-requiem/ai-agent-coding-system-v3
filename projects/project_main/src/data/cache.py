import json
import os
import pathlib
import re
import tempfile
import time
from typing import Any, Optional


class Cache:
    """
    A disk-based cache with TTL (time-to-live) support.
    
    Args:
        root: The root directory where cache files will be stored
        ttl_seconds: Time in seconds after which cache entries expire (default: 300.0)
    """
    
    def __init__(self, root: pathlib.Path, ttl_seconds: float = 300.0) -> None:
        self.root = pathlib.Path(root)
        self.ttl_seconds = ttl_seconds
        # Ensure the root directory exists
        self.root.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_key(self, key: str) -> str:
        """
        Sanitize a key to create a safe filename.
        
        Args:
            key: The original key string
            
        Returns:
            A sanitized filename ending with .json
        """
        if not key:
            key = "default"
        
        # Replace any character not in [A-Za-z0-9_-.] with '_'
        sanitized = re.sub(r'[^A-Za-z0-9_\-.]', '_', key)
        
        # If result is empty after sanitization, use 'default'
        if not sanitized:
            sanitized = "default"
        
        # Ensure the filename ends with .json (remove any existing .json suffix to avoid duplication)
        if sanitized.endswith('.json'):
            sanitized = sanitized[:-5]
        
        return sanitized + '.json'
    
    def path_for(self, key: str) -> pathlib.Path:
        """
        Get the filesystem path for a given key.
        
        Args:
            key: The cache key
            
        Returns:
            Path object pointing to the cache file for this key
        """
        filename = self._sanitize_key(key)
        return self.root / filename
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key
            value: The value to store (must be JSON-serializable)
            
        Raises:
            TypeError: If the value is not JSON-serializable
            OSError: If there's an issue writing to the filesystem
        """
        # Ensure the root directory exists
        self.root.mkdir(parents=True, exist_ok=True)
        
        path = self.path_for(key)
        
        # Serialize the value to JSON
        try:
            json_data = json.dumps(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Value is not JSON-serializable: {e}")
        
        # Write atomically using a temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            dir=self.root, 
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_file.write(json_data)
            tmp_path = pathlib.Path(tmp_file.name)
        
        # Atomically replace the target file
        os.replace(tmp_path, path)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value if found and not expired, None otherwise
        """
        path = self.path_for(key)
        
        # Check if file exists
        if not path.exists():
            return None
        
        # Check if file is expired
        try:
            mtime = path.stat().st_mtime
            current_time = time.time()
            
            if (current_time - mtime) > self.ttl_seconds:
                # File is expired, optionally delete it
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
        except OSError:
            return None
        
        # Read and parse the JSON file
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # File is corrupted, delete it and return None
            try:
                path.unlink()
            except OSError:
                pass
            return None
