import importlib.util, pytest
if importlib.util.find_spec('src.slatepress.cli') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.slatepress.cli import main

def test_main_success_default_output(tmp_path):
    source_dir = tmp_path / "content"
    source_dir.mkdir()
    
    result = main([str(source_dir)])
    assert result == 0
    assert (source_dir / "_site").exists()

def test_main_success_custom_output(tmp_path):
    source_dir = tmp_path / "content"
    source_dir.mkdir()
    output_dir = tmp_path / "custom_site"
    
    result = main([str(source_dir), str(output_dir)])
    assert result == 0
    assert output_dir.exists()

def test_main_source_not_exists(tmp_path):
    non_existent = tmp_path / "does_not_exist"
    
    result = main([str(non_existent)])
    assert result == 2