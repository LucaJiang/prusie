"""Shared offline presentation tokens; importing them needs only the standard library.

Statistics are never rounded in exported data. ``format_number`` is for labels
and prose only. Plot functions apply ``PLOT_STYLE`` in a local rc_context; pass
``style=False`` to use the caller's current style or a dict to override tokens.
"""

from __future__ import annotations

import math
import numbers

COLORS = {
    "text": "#202b33",
    "muted": "#596774",
    "grid": "#e6ebef",
    "background": "#ffffff",
    "before": "#3b6e99",
    "after": "#b45d20",
    "flagged": "#a53e52",
    "conditional": "#80659d",
    "missing": "#c4cbd1",
    "not_flagged": "#526b78",
    "accepted": "#28766b",
    "rejected": "#79858e",
    "cs": ("#3974a4", "#c47728", "#328575", "#8662a4", "#ad536d", "#6f7e2d"),
    "sequential": "viridis",
    "diverging": "coolwarm",
    "pip_cmap": "Blues",
    "ld_r2_cmap": "Reds",
    "ld_signed_cmap": "RdBu_r",
}

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 10,
    "figure.titlesize": 13,
    "figure.titleweight": "semibold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "legend.frameon": False,
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.edgecolor": COLORS["muted"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "axes.titlepad": 8,
    "axes.labelpad": 5,
    "savefig.dpi": 150,
    "svg.fonttype": "none",
}


def format_number(value, *, probability=False, precision=4):
    """Format display values without turning interior probabilities into 0/1.

    Exact endpoints remain ``0``/``1``. Tiny nonzero values use scientific
    notation. A rounded-up probability near one is shown as a strict lower
    bound. JSON/CSV/NPZ exports retain the original value separately.
    """
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, numbers.Real):
        return str(value)
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
        raise ValueError("precision must be a positive integer")
    number = float(value)
    if math.isnan(number):
        return "NA"
    if math.isinf(number):
        return "∞" if number > 0 else "−∞"
    if number == 0:
        return "0"
    if probability and 0 < number < 1 and format(number, f".{precision}g") == "1":
        boundary = 1 - 10.0 ** (-precision)
        return f"> {boundary:.{precision}f}"
    return format(number, f".{precision}g")


def format_pip_cell(value):
    "Short heatmap labels; full-precision probabilities stay in plotted data."
    if value is None or not math.isfinite(float(value)):
        return "NA"
    number = float(value)
    if number in (0.0, 1.0):
        return str(int(number))
    if 0.0 < number < 0.01:
        return "<0.01"
    if 0.99 < number < 1.0:
        return ">0.99"
    return f"{number:.2f}"


def format_ld_cell(value):
    "Four-decimal LD labels that do not turn near-unit LD into exact one."
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    if number in (-1.0, 0.0, 1.0):
        return str(int(number))
    if 0.9999 < number < 1.0:
        return ">0.9999"
    if -1.0 < number < -0.9999:
        return "<-0.9999"
    if 0.0 < abs(number) < 0.0001:
        return f"{number:.2g}"
    return f"{number:.4f}"


def annotation_color(rgba, *, background=(1.0, 1.0, 1.0, 1.0)):
    "Choose black/white by contrast against the actual mapped cell color."
    alpha = float(rgba[3]) if len(rgba) > 3 else 1.0
    rgb = [
        alpha * float(c) + (1.0 - alpha) * float(b)
        for c, b in zip(rgba[:3], background[:3], strict=True)
    ]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    luminance = sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))
    return "#000000" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "#ffffff"


REPORT_CSS = """
:root{--text:#202b33;--muted:#596774;--line:#dce3e8;--panel:#f5f7f9;
 --before:#3b6e99;--after:#b45d20;--flagged:#a53e52;--missing:#c4cbd1}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{max-width:1160px;margin:auto;
 padding:24px 30px;color:var(--text);background:white;font:16px/1.55 system-ui,-apple-system,
 BlinkMacSystemFont,"Segoe UI",sans-serif}h1{font-size:1.9rem;line-height:1.2;margin:.3em 0 .6em}
 h2{font-size:1.4rem;line-height:1.3;margin:2em 0 .6em;padding-bottom:.25em;border-bottom:1px solid var(--line)}
 h3{font-size:1.1rem;margin:1.3em 0 .4em}p{margin:.65em 0}a{color:#28618b;text-underline-offset:3px}
 nav{padding:12px 0;border-block:1px solid var(--line);margin:1.2rem 0}nav a{display:inline-block;margin:3px 16px 3px 0}
 .muted,.caption,figcaption{color:var(--muted)}code,kbd,pre{font-family:ui-monospace,SFMono-Regular,
 Consolas,"Liberation Mono",monospace;font-size:.88em}code{overflow-wrap:anywhere}pre{padding:14px 16px;
 background:var(--panel);white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}
 figure{margin:1.6rem 0}figure img{display:block;max-width:100%;height:auto;margin:auto}
 figcaption{font-size:.9rem;margin:.6em 0}table{border-collapse:collapse;width:100%;font-size:.9rem;
 font-variant-numeric:tabular-nums}thead{background:var(--panel)}th,td{padding:8px 10px;
 border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600;white-space:nowrap}
 td.numeric{text-align:right}td{overflow-wrap:anywhere}.table-scroll{overflow-x:auto;margin:.8rem 0 1.2rem}
 details{margin:1rem 0;padding:.6rem 0;border-bottom:1px solid var(--line)}summary{cursor:pointer;font-weight:600}
 details[open]>summary{margin-bottom:.6rem}.status-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
 gap:12px;list-style:none;padding:0}.status-list li{padding:10px 12px;border-left:3px solid var(--line)}
 .status-label{font-weight:600;display:block}.exact-value{cursor:help;text-decoration:underline dotted var(--muted)}
 @media(max-width:800px){body{padding:16px;font-size:16px}h1{font-size:1.6rem}h2{font-size:1.25rem}
 figure{margin:1.2rem -6px}.table-scroll{max-width:100%}th,td{padding:7px 8px}nav a{margin-right:12px}}
 @media print{body{max-width:none;padding:0;font-size:10pt;color:black}nav{display:none}h2{break-after:avoid}
 figure,table{break-inside:avoid}a{color:inherit}details{border:0}.table-scroll{overflow:visible}pre{background:white}}
"""  # noqa: E501 - Preserve rendered HTML/CSS bytes.
