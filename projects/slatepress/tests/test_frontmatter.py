import importlib.util, pytest
if importlib.util.find_spec('src.slatepress.frontmatter') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.slatepress.frontmatter import split_front_matter, parse_front_matter

def test_split_front_matter_happy_path():
    text = "---\ntitle: Hello World\n---\nBody content here."
    fm, body = split_front_matter(text)
    assert fm == "title: Hello World"
    assert body == "Body content here."

def test_split_front_matter_no_front_matter():
    text = "Just regular body text without front matter."
    fm, body = split_front_matter(text)
    assert fm is None
    assert body == "Just regular body text without front matter."

def test_split_front_matter_unclosed_delimiter():
    text = "---\ntitle: Unclosed\nBody content."
    with pytest.raises(ValueError):
        split_front_matter(text)

def test_parse_front_matter_happy_path():
    text = "title: My Post\n# This is a comment\nauthor: Jane Doe"
    result = parse_front_matter(text)
    assert result == {
        "title": "My Post",
        "author": "Jane Doe"
    }

def test_parse_front_matter_empty():
    result = parse_front_matter("")
    assert result == {}