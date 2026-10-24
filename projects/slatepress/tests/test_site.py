import importlib.util
import pytest

if importlib.util.find_spec("src.slatepress.site") is None:
    pytest.skip("implementation not written yet", allow_module_level=True)

from pathlib import Path
from src.slatepress.site import build_site


def test_build_site_basic(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    md_file = source_dir / "index.md"
    md_file.write_text("# Hello World\n", encoding="utf-8")

    asset_file = source_dir / "style.css"
    asset_file.write_text("body { color: red; }", encoding="utf-8")

    written_paths = build_site(source_dir, output_dir)

    assert isinstance(written_paths, list)
    assert len(written_paths) == 2
    assert (output_dir / "index.html").exists()
    assert (output_dir / "style.css").exists()
    assert str(output_dir / "index.html") in written_paths
    assert str(output_dir / "style.css") in written_paths


def test_build_site_skips_hidden_and_underscore(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    hidden_file = source_dir / ".hidden.md"
    hidden_file.write_text("hidden", encoding="utf-8")

    underscore_file = source_dir / "_template.md"
    underscore_file.write_text("template", encoding="utf-8")

    hidden_dir = source_dir / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config.md").write_text("config", encoding="utf-8")

    underscore_dir = source_dir / "_partials"
    underscore_dir.mkdir()
    (underscore_dir / "header.md").write_text("header", encoding="utf-8")

    valid_file = source_dir / "about.md"
    valid_file.write_text("# About", encoding="utf-8")

    written_paths = build_site(source_dir, output_dir)

    assert len(written_paths) == 1
    assert (output_dir / "about.html").exists()
    assert not (output_dir / ".hidden.html").exists()
    assert not (output_dir / "_template.html").exists()
    assert not (output_dir / "config.html").exists()
    assert not (output_dir / "_partials").exists()


def test_build_site_nested_directories(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    
    blog_dir = source_dir / "blog"
    blog_dir.mkdir(parents=True)
    
    post_file = blog_dir / "post1.md"
    post_file.write_text("# Post 1", encoding="utf-8")

    written_paths = build_site(source_dir, output_dir)

    assert (output_dir / "blog" / "post1.html").exists()
    assert str(output_dir / "blog" / "post1.html") in written_paths