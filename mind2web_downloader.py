# Multimodal Mind2web download script for the test_website slice only.
from datasets import load_dataset
from bs4 import BeautifulSoup, NavigableString, Tag

if __name__ != "__main__":
    ds = None   # importing this module only to use html_to_axtree — no download needed
else:
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split = 'test_website')


# ── AXTree generation ─────────────────────────────────────────────────────────
#
# Each node in the output tree has a unique object ID and indentation that
# mirrors the exact DOM hierarchy, e.g.:
#
#   [#1] (main)
#     [#2] (banner)
#       [#3] (heading) level=1 | "Site Title"
#       [#4] (navigation)
#         [#5] (link) href="/home" | "Home"
#         [#6] (link) href="/about" | "About"
#     [#7] (form)
#       [#8] (textbox) placeholder="Search" required | ""
#       [#9] (button) | "Submit"
#
# Every interactive element (links, buttons, inputs, selects) is always emitted
# even if it has no accessible name, so candidate actions can be correlated with
# tree nodes by their ID.

# Tags that map to an explicit ARIA role
_TAG_TO_ROLE = {
    "a":        "link",
    "button":   "button",
    "input":    "textbox",      # refined below by input[type]
    "select":   "combobox",
    "option":   "option",
    "textarea": "textbox",
    "img":      "image",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "nav":      "navigation",
    "main":     "main",
    "header":   "banner",
    "footer":   "contentinfo",
    "form":     "form",
    "table":    "table",
    "thead":    "rowgroup",
    "tbody":    "rowgroup",
    "tr":       "row",
    "td":       "cell",
    "th":       "columnheader",
    "ul":       "list",
    "ol":       "list",
    "li":       "listitem",
    "label":    "label",
    "dialog":   "dialog",
    "details":  "group",
    "summary":  "button",
    "menu":     "menu",
    "menuitem": "menuitem",
}

_INPUT_TYPE_TO_ROLE = {
    "checkbox": "checkbox",
    "radio":    "radio",
    "submit":   "button",
    "button":   "button",
    "reset":    "button",
    "search":   "searchbox",
    "range":    "slider",
    "number":   "spinbutton",
}

# Tags whose subtrees are entirely skipped
_SKIP_TAGS = frozenset({
    "script", "style", "meta", "link", "noscript",
    "svg", "path", "defs", "symbol", "use", "g",
    "head", "iframe", "canvas",
})

# Layout tags that fall back to role="generic" when no explicit role is set
_LAYOUT_TAGS = frozenset({"div", "span", "section", "article", "aside", "p"})


def _accessible_name(el: Tag) -> str:
    """Return the best accessible name for an element, empty string if none."""
    for attr in ("aria-label", "aria-labelledby", "title"):
        v = (el.get(attr) or "").strip()
        if v:
            return v[:150]
    if el.name == "img":
        v = (el.get("alt") or "").strip()
        if v:
            return v[:150]
    if el.name in ("input", "textarea"):
        for attr in ("value", "placeholder"):
            v = (el.get(attr) or "").strip()
            if v:
                return v[:150]
    # Shallow text (direct text children only, to avoid pulling in subtree noise)
    parts = [c.strip() for c in el.children
              if isinstance(c, NavigableString) and c.strip()]
    text = " ".join(parts)
    if not text:
        # Fall back to full subtree text for leaves
        text = el.get_text(separator=" ", strip=True)
    return text[:150]


def _node_attributes(el: Tag, role: str) -> list[str]:
    """Return a list of key=value attribute strings for a node line."""
    attrs = []

    # Heading level
    if role == "heading" and el.name and el.name[1:].isdigit():
        attrs.append(f"level={el.name[1]}")

    # Boolean states
    if el.get("checked") is not None:
        attrs.append("checked")
    if el.get("selected") is not None:
        attrs.append("selected")
    if el.get("disabled") is not None:
        attrs.append("disabled")
    if el.get("required") is not None:
        attrs.append("required")
    if el.get("readonly") is not None:
        attrs.append("readonly")
    if el.get("multiple") is not None:
        attrs.append("multiple")

    # Input type (when non-default)
    if el.name == "input":
        itype = (el.get("type") or "text").lower()
        if itype != "text":
            attrs.append(f"type={itype}")

    # href (truncated)
    if el.name == "a" and el.get("href"):
        href = el["href"].strip()
        if href and href != "#":
            attrs.append(f'href="{href[:80]}"')

    # DOM id (gives candidate action correlation anchor)
    dom_id = el.get("id") or el.get("data-id") or el.get("data-testid")
    if dom_id:
        attrs.append(f'id="{str(dom_id)[:40]}"')

    return attrs


def html_to_axtree(
    html: str,
    max_nodes: int = 300,
    indent: str = "  ",
) -> str:
    """
    Convert a cleaned HTML string into an accessibility tree with object IDs.

    Parameters
    ----------
    html      : cleaned HTML string (e.g. example["cleaned_html"])
    max_nodes : maximum number of nodes to emit before truncating
    indent    : indentation string per depth level

    Returns
    -------
    Multi-line string representing the AXTree.
    Each node line format:
        <indent>[#<id>] (<role>) <attrs> | "<accessible name>"
    Container nodes with no name emit the pipe but leave the name blank.
    """
    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []
    counter = [0]   # mutable counter shared across recursive calls

    def _visit(el: Tag, depth: int) -> None:
        if counter[0] >= max_nodes:
            return
        if not isinstance(el, Tag):
            return
        tag = el.name.lower() if el.name else ""
        if tag in _SKIP_TAGS:
            return

        # Determine role
        role = el.get("role") or _TAG_TO_ROLE.get(tag)
        if role is None:
            if tag in _LAYOUT_TAGS:
                # layout tag with no explicit role → emit as "generic"
                role = "generic"
            else:
                # unknown tag (e.g. custom elements) → emit as "generic"
                role = "generic"

        # Refine input[type] role
        if tag == "input":
            itype = (el.get("type") or "text").lower()
            role = _INPUT_TYPE_TO_ROLE.get(itype, "textbox")

        name = _accessible_name(el)

        counter[0] += 1
        node_id = counter[0]
        attr_list = _node_attributes(el, role)
        attr_str = (" " + " ".join(attr_list)) if attr_list else ""
        name_str = f'"{name}"' if name else '""'
        prefix = indent * depth
        lines.append(f"{prefix}[#{node_id}] ({role}){attr_str} | {name_str}")

        for child in el.children:
            _visit(child, depth + 1)

    root = soup.find("body") or soup
    for child in root.children:
        _visit(child, 0)

    if counter[0] >= max_nodes:
        lines.append(f"... [{counter[0] - max_nodes} additional nodes truncated] ...")

    return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assert ds is not None
    example = ds[0]
    tree = html_to_axtree(example["cleaned_html"], max_nodes=300)
    print(f"Task : {example['confirmed_task']}")
    print(f"Nodes: {len(tree.splitlines())}")
    print()
    print(tree)
