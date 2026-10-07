"""Native SuSiE-RSS inference, referenced to pinned official susieR 0.16.6."""
from .api import susie_rss, susie_suff_stat
from .result import SusieResult
from .datasets import load_example

__all__ = ["susie_rss", "susie_suff_stat", "SusieResult", "load_example"]
__version__ = "0.2.3rc6"
