"""Native SuSiE-RSS inference, referenced to pinned official susieR 0.16.6."""
from .api import susie_rss, susie_suff_stat
from .result import SusieResult

__all__ = ["susie_rss", "susie_suff_stat", "SusieResult"]
__version__ = "0.2.3rc4"
