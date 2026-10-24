import importlib.util, pytest
if importlib.util.find_spec('src.slatepress.markdown') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.slatepress.markdown import inline_markup


def test_inline_markup_plain_text():
    text = "Hello world"
    assert inline_markup(text) == "Hello world"


def test_inline_markup_bold():
    text = "This is **bold** text"
    assert inline_markup(text) == "This is <strong>bold</strong> text"


def test_inline_markup_italic():
    text = "This is *italic* text"
    assert inline_markup(text) == "This is <em>italic</em> text"


def test_inline_markup_code():
    text = "This is `code` text"
    assert inline_markup(text) == "This is <code>code</code> text"


def test_inline_markup_html_escaping_first():
    text = "Use <script> & alert('hack')</script> with **bold**"
    result = inline_markup(text)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "&amp;" in result
    assert "<strong>bold</strong>" in result


def test_inline_markup_multiple_elements():
    text = "**Bold**, *italic*, and `code`."
    assert inline_markup(text) == "<strong>Bold</strong>, <em>italic</em>, and <code>code</code>."