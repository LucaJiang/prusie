"""Common plotting helpers."""

from __future__ import annotations

import functools
import re
import textwrap

import numpy as np

from .._style import COLORS, PLOT_STYLE, format_number


def _local_style(function):
    @functools.wraps(function)
    def decorated(*args, style=None, **kwargs):
        import matplotlib.pyplot as plt

        local = {} if style is False else dict(PLOT_STYLE)
        if style is not None and style is not False:
            local.update(style)
        if "backend" in local:
            raise ValueError(
                "Select the Matplotlib backend before plotting; backend is not a local style option"
            )
        with plt.rc_context(local):
            return function(*args, **kwargs)

    return decorated


def _axes(axes, count, *, figsize=None, height_ratios=None):
    import matplotlib.pyplot as plt

    if axes is None:
        figure, supplied = plt.subplots(
            count,
            1,
            squeeze=False,
            figsize=figsize or (9, 2.6 * count),
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
        )
        return figure, supplied[:, 0]
    supplied = np.asarray(axes, dtype=object).reshape(-1)
    if len(supplied) != count:
        raise ValueError(f"This view requires {count} axes")
    if len({id(ax.figure) for ax in supplied}) != 1:
        raise ValueError("All axes must belong to one figure")
    return supplied[0].figure, supplied


def _missing(ax, message):
    ax.text(
        0.5,
        0.5,
        textwrap.fill(message, 70),
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=10,
    )


def _layout(value):
    if value not in ("full", "compact"):
        raise ValueError("layout must be 'full' or 'compact'")
    return value == "compact"


def _component_color(component_id):
    return COLORS["cs"][int(component_id) % len(COLORS["cs"])]


def _bottom_labels(axes, label):
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel(label)


def _table(ax, rows, labels, *, page=0, page_size=None):
    """Draw one page of a table; rows are never truncated silently."""
    if not isinstance(page, int) or page < 0:
        raise ValueError("table_page must be a nonnegative integer")
    if page_size is not None and (not isinstance(page_size, int) or page_size < 1):
        raise ValueError("table_page_size must be a positive integer or None")
    if page_size is None and page:
        raise ValueError("table_page requires table_page_size")
    start = page * page_size if page_size else 0
    selected = rows[start : start + page_size] if page_size else rows
    if rows and not selected:
        raise ValueError("table_page is outside the available records")

    def display(value):
        if isinstance(value, (float, np.floating)):
            return format_number(value)
        text = "NA" if value is None else str(value)
        text = text if len(text) <= 48 else text[:45] + "…"
        return "\n".join(textwrap.wrap(text, width=24 if len(labels) >= 5 else 36))

    shown = [[display(v) for v in row] for row in selected]
    table = ax.table(cellText=shown, colLabels=labels, cellLoc="left", bbox=[0, 0.14, 1, 0.84])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Multi-line cells need proportionate row heights; wrapping alone would
    # move text into the next row. Keep the complete table within its own axes.
    line_counts = [1] + [max(text.count("\n") + 1 for text in row) for row in shown]
    for (row, _), cell in table.get_celld().items():
        cell.set_height(line_counts[row] / sum(line_counts))
    if page_size and len(rows) > page_size:
        ax.text(
            0.5,
            0.03,
            (
                f"Rows {start + 1}"
                f"–{start + len(selected)}"
                f" of {len(rows)}"
                f"; "
                "select table_page to view others; complete records in returned data."
            ),
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=9,
        )
    return table


def _number(value):
    if isinstance(value, dict) and "nonfinite" in value:
        return float(value["nonfinite"])
    return np.nan if value is None else float(value)


def _display_labels(labels):
    """Shorten long labels in the middle, lengthening prefixes to keep them distinct."""
    full = [str(label) for label in labels]
    shown = [value if len(value) <= 30 else value[:15] + "…" + value[-10:] for value in full]
    for i, value in enumerate(full):
        if len(value) > 30 and shown.count(shown[i]) > 1:
            prefix = 16
            while prefix < len(value) - 10:
                candidate = value[:prefix] + "…" + value[-10:]
                if candidate not in shown:
                    shown[i] = candidate
                    break
                prefix += 1
            else:
                shown[i] = value
    return shown


def _variant_ticks(ax, labels, *, axis="x", priority_indices=()):
    """Choose a deterministic subset of tick labels; all observations remain plotted."""
    limit = 6 if any(len(str(label)) > 30 for label in labels) else 12
    priority = list(dict.fromkeys(int(i) for i in priority_indices if 0 <= i < len(labels)))[:limit]
    other = (
        np.linspace(0, len(labels) - 1, min(limit - len(priority), len(labels)), dtype=int)
        if labels
        else []
    )
    indices = (
        list(range(len(labels))) if len(labels) <= limit else sorted(set(priority) | set(other))
    )
    display = _display_labels(labels)
    selected = [display[i] for i in indices]
    if axis == "x":
        ax.set_xticks(indices, selected, rotation=60, ha="right")
    else:
        ax.set_yticks(indices, selected)
    return list(indices)


def _ld(R, p):
    matrix = np.asarray(R, dtype=float)
    if matrix.shape != (p, p) or not np.all(np.isfinite(matrix)):
        raise ValueError("R must be a finite matrix aligned to the variant table")
    return matrix


def _chromosome(value):
    if value is None or str(value).lower() in ("", "unknown", "not_assessed"):
        return None
    return re.sub(r"^chr", "", str(value), flags=re.IGNORECASE)
