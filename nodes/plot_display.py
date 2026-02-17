"""Matplotlib live plot display sink node."""
import numpy as np
import matplotlib.pyplot as plt

from sigflow.node import sink_node
from sigflow.types import Port, TimeSeries1D


@sink_node(
    name="plot_display",
    inputs=[Port("signal", TimeSeries1D)],
    category="display",
)
def plot_display(item, *, state, config):
    if "fig" not in state:
        plt.ion()
        state["fig"], state["ax"] = plt.subplots()
        state["line"] = None

    ax = state["ax"]
    data = item.data

    if state["line"] is None:
        state["line"], = ax.plot(data)
    else:
        state["line"].set_ydata(data)
        if len(data) != len(state["line"].get_xdata()):
            state["line"].set_xdata(np.arange(len(data)))
            state["line"].set_ydata(data)
            ax.relim()
            ax.autoscale_view()

    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()


@plot_display.cleanup
def plot_display_cleanup(state, config):
    if "fig" in state:
        plt.close(state["fig"])
