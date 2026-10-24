import importlib.util, pytest
if importlib.util.find_spec('src.slatepress.html') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.slatepress.html import escape, page

def test_escape_basic():
    assert escape("Hello & Welcome") == "Hello &amp; Welcome"
    assert escape("3 < 5") == "3 &lt; 5"
    assert escape("5 > 3") == "5 &gt; 3"

def test_escape_combined():
    assert escape("<span>Click & Save</span>") == "&lt;span&gt;Click &amp; Save&lt;/span&gt;"

def test_escape_no_special_chars():
    assert escape("Plain text 123") == "Plain text 123"

def test_page_basic():
    html_output = page("My Title", "<p>Hello World</p>")
    assert "<title>My Title</title>" in html_output
    assert "<main><p>Hello World</p></main>" in html_output
    assert "<!DOCTYPE html>" in html_output
    assert "stylesheet" not in html_output

def test_page_with_css():
    html_output = page("My Title", "<p>Hello World</p>", css_link="style.css")
    assert "<title>My Title</title>" in html_output
    assert "<main><p>Hello World</p></main>" in html_output
    assert 'href="style.css"' in html_output
    assert "<link" in html_output