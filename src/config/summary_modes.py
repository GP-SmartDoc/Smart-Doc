from enum import Enum

class SummaryMode(Enum):
    SNAPSHOT = "snapshot"
    OVERVIEW = "overview"
    DEEPDIVE = "deepdive"


MODE_CONFIG = {
    SummaryMode.SNAPSHOT: {
        "compression": 0.02,
        "max_tokens": 80,
        "detail": 1
    },
    SummaryMode.OVERVIEW: {
        "compression": 0.12,
        "max_tokens": 300,
        "detail": 2
    },
    SummaryMode.DEEPDIVE: {
        "compression": 0.35,
        "max_tokens": 900,
        "detail": 3
    }
}
