"""
Utilities for lightly pruning HTML before sending it to an LLM.

The default pass is intentionally conservative:
  - remove clearly non-functional markup like script/style/template
  - drop hidden elements and comments
  - unwrap trivial presentational tags
  - strip noisy attributes while keeping structural and interactive ones
  - remove empty, non-semantic wrappers
"""

from __future__ import annotations

import re
from bs4 import BeautifulSoup, Comment, NavigableString, Tag


DROP_WITH_CONTENT = frozenset({
    "script",
    "style",
    "noscript",
    "template",
    "meta",
    "link",
    "base",
})

PRESENTATIONAL_WRAPPERS = frozenset({
    "b",
    "strong",
    "i",
    "em",
    "u",
    "font",
})

SEMANTIC_OR_INTERACTIVE = frozenset({
    "a",
    "button",
    "input",
    "select",
    "option",
    "textarea",
    "label",
    "form",
    "fieldset",
    "legend",
    "main",
    "nav",
    "header",
    "footer",
    "section",
    "article",
    "aside",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "ul",
    "ol",
    "li",
    "details",
    "summary",
    "dialog",
    "img",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
})

KEEP_EXACT_ATTRIBUTES = frozenset({
    "id",
    "class",
    "role",
    "href",
    "src",
    "alt",
    "title",
    "name",
    "type",
    "value",
    "placeholder",
    "for",
    "action",
    "method",
    "selected",
    "checked",
    "disabled",
    "readonly",
    "required",
    "multiple",
    "tabindex",
    "aria-label",
    "aria-labelledby",
    "aria-describedby",
    "aria-controls",
    "aria-expanded",
    "aria-checked",
    "aria-selected",
    "aria-current",
    "aria-pressed",
    "aria-hidden",
    "data-testid",
    "data-id",
})

DROP_STYLE_PATTERNS = (
    "display:none",
    "display: none",
    "visibility:hidden",
    "visibility: hidden",
)


def _has_useful_attrs(tag: Tag) -> bool:
    for attr_name in tag.attrs:
        if attr_name in KEEP_EXACT_ATTRIBUTES:
            return True
        if attr_name.startswith("aria-"):
            return True
        if attr_name.startswith("data-") and "test" in attr_name:
            return True
    return False


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    if str(tag.get("aria-hidden", "")).strip().lower() == "true":
        return True
    if tag.name == "input" and str(tag.get("type", "")).strip().lower() == "hidden":
        return True

    style = str(tag.get("style", "")).strip().lower()
    return any(pattern in style for pattern in DROP_STYLE_PATTERNS)


def _normalize_text_nodes(root: Tag | BeautifulSoup) -> None:
    for node in list(root.descendants):
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue

        parent = node.parent
        if parent and parent.name in {"pre", "code", "textarea"}:
            continue

        collapsed = re.sub(r"\s+", " ", str(node))
        if collapsed.strip():
            node.replace_with(collapsed)
        else:
            if node.previous_sibling is not None and node.next_sibling is not None:
                node.replace_with(" ")
            else:
                node.extract()

    if isinstance(root, Tag):
        root.smooth()
    else:
        for child in root.find_all(True):
            child.smooth()


def _strip_attributes(root: Tag | BeautifulSoup) -> None:
    for tag in root.find_all(True):
        kept = {}
        for attr_name, value in tag.attrs.items():
            if attr_name in KEEP_EXACT_ATTRIBUTES or attr_name.startswith("aria-"):
                kept[attr_name] = value
            elif attr_name.startswith("data-") and "test" in attr_name:
                kept[attr_name] = value
        tag.attrs = kept


def _remove_empty_wrappers(root: Tag | BeautifulSoup) -> None:
    changed = True
    while changed:
        changed = False
        for tag in list(root.find_all(True)):
            if tag.name in SEMANTIC_OR_INTERACTIVE:
                continue
            if _has_useful_attrs(tag):
                continue
            if tag.find(True) is not None:
                continue
            if tag.get_text(strip=True):
                continue
            tag.decompose()
            changed = True


def prune_html_dom(
    html: str,
    *,
    keep_body_only: bool = True,
    strip_attributes: bool = True,
    unwrap_presentational: bool = True,
    remove_hidden: bool = True,
    remove_empty_wrappers: bool = True,
) -> str:
    """
    Return a lightly pruned HTML DOM string.

    This pass is aimed at preserving navigational and structural cues while
    cutting obvious token-heavy noise.
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    if keep_body_only and soup.body is not None:
        if soup.head is not None:
            soup.head.decompose()
        root: Tag | BeautifulSoup = soup.body
    else:
        root = soup

    for tag in list(root.find_all(DROP_WITH_CONTENT)):
        tag.decompose()

    if remove_hidden:
        for tag in list(root.find_all(True)):
            if _is_hidden(tag):
                tag.decompose()

    if unwrap_presentational:
        for tag in list(root.find_all(PRESENTATIONAL_WRAPPERS)):
            if not _has_useful_attrs(tag):
                tag.unwrap()

    if strip_attributes:
        _strip_attributes(root)

    _normalize_text_nodes(root)

    if remove_empty_wrappers:
        _remove_empty_wrappers(root)

    return str(root)
