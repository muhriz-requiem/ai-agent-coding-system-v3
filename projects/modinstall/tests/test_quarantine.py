import importlib.util, pytest
if importlib.util.find_spec('src.installer.quarantine') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.installer.quarantine import verify_artifact

import hashlib
import tempfile
import os

def test_verify_artifact_correct_hash():
    content = b"test artifact content"
    correct_hash = hashlib.sha256(content).hexdigest()
    
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(content)
        temp_path = tf.name
        
    try:
        result = verify_artifact(temp_path, correct_hash)
        assert result is True
    finally:
        os.unlink(temp_path)

def test_verify_artifact_mismatch_hash():
    content = b"test artifact content"
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(content)
        temp_path = tf.name
        
    try:
        result = verify_artifact(temp_path, wrong_hash)
        assert result is False
    finally:
        os.unlink(temp_path)

def test_verify_artifact_nonexistent_file():
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    non_existent_path = "/non/existent/path/to/artifact.bin"
    
    # Should not raise an exception, just return False
    try:
        result = verify_artifact(non_existent_path, wrong_hash)
        assert result is False
    except Exception as e:
        pytest.fail(f"verify_artifact raised an exception on non-existent file: {e}")