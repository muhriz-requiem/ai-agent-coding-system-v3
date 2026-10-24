import importlib.util, pytest
if importlib.util.find_spec('src.slatepress') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.slatepress import (
    split_front_matter,
    parse_front_matter,
    inline_markup,
    blocks,
    escape,
    page,
    build_site,
    main,
)

def test_split_front_matter_reexported():
    assert callable(split_front_matter)

def test_parse_front_matter_reexported():
    assert callable(parse_front_matter)

def test_inline_markup_reexported():
    assert callable(inline_markup)

def test_blocks_reexported():
    assert callable(blocks)

def test_escape_reexported():
    assert callable(escape)

def test_page_reexported():
    assert callable(page)

def test_build_site_reexported():
    assert callable(build_site)

def test_main_reexported():
    assert callable(main)