"""
Convenience data containers
"""


from collections import namedtuple


RotationResult = namedtuple(
    "RotationResult",
    ["start", "i_start", "rotated","angle", "score"]
)

