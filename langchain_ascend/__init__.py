"""langchain-ascend-affinity: openJiuwen agent-core compute affinity, ported to LangChain."""

from langchain_ascend.llms.chat_ascend import AscendAffinityChatModel
from langchain_ascend.prefix_tracker import PrefixCacheTracker, ReleasePlan

__version__ = "0.2.0"

__all__ = [
    "AscendAffinityChatModel",
    "PrefixCacheTracker",
    "ReleasePlan",
    "__version__",
]
