"""Flip process node — vertical and/or horizontal."""
from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D


@process_node(
    name="flip",
    inputs=[Port("frame", TimeSeries2D)],
    outputs=[Port("frame", TimeSeries2D)],
    category="transform",
    params=[
        Param("vertical", "bool", False),
        Param("horizontal", "bool", True),
    ],
)
def flip(item, *, state, config):
    data = item.data
    if config["vertical"]:
        data = data[::-1]
    if config["horizontal"]:
        data = data[:, ::-1]
    return {"frame": item.replace(data=data)}
