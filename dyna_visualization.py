"""Plotly-Abbildungen der Dynasearch-Demo: Karten, Verlauf, Budget-/Größen-Vergleich, Sweeps, Streuung, Skalierung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import dyna_constants as C

TOUR_COLOR = "#4c78a8"
DYNA_COLOR = "#54a24b"
SEQ_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _line_trace(xy, edges, color, name, dash=None, width=2.5, showlegend=True):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_tour(xy, tour, ghost=None, changed_nodes=None):
    """Tour; `ghost` (optional) ist eine zweite Tour, die blass darunter gezeichnet wird. `changed_nodes` (optional)
    hebt die von einem Zug betroffenen Knoten hervor (zeigt bei Dynasearch mehrere getrennte Stellen auf einmal)."""
    fig = go.Figure()
    if ghost is not None:
        fig.add_trace(_line_trace(xy, _tour_edges_list(ghost), "#c9d6e6", "vorige Tour", width=6))
    fig.add_trace(_line_trace(xy, _tour_edges_list(tour), TOUR_COLOR, "aktuelle Tour"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=6, color="white", line=dict(width=1.5, color=TOUR_COLOR)), showlegend=False, hoverinfo="skip"))
    if changed_nodes:
        idx = np.asarray(sorted(changed_nodes))
        idx = idx[idx > 0]
        if len(idx):
            fig.add_trace(go.Scatter(x=xy[idx, 0], y=xy[idx, 1], mode="markers", marker=dict(size=11, color=DYNA_COLOR, line=dict(width=1.5, color="white")), name="geänderte Knoten"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_trace(trace_iter, trace_length, bound, mode_label, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace_iter, y=trace_length, mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=6), name=mode_label))
    fig.add_hline(y=bound, line=dict(color="#7f7f7f", dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.update_xaxes(title_text="Iteration (voller Rescan)")
    fig.update_yaxes(title_text="Länge (km)")
    return _base(fig, 320)


def build_moves_per_iteration(moves_per_step, x=None, x_title="Dynasearch-Iteration"):
    """Zahl der GLEICHZEITIG angewandten, unabhängigen Züge je Dynasearch-Iteration - zeigt konkret, wie viele
    Kombinationen die DP in einem Durchgang statt einzeln nacheinander anwendet. Mit `x`/`x_title` auch für andere
    Größen auf der x-Achse nutzbar (z. B. Instanzgröße im Effizienz-Experiment)."""
    xs = list(x) if x is not None else list(range(1, len(moves_per_step) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=moves_per_step, marker_color=DYNA_COLOR, name="kombinierte Züge"))
    fig.update_xaxes(title_text=x_title, type="category" if x is not None else "linear")
    fig.update_yaxes(title_text="Ø gleichzeitig angewandte Züge")
    return _base(fig, 280)


def build_budget(rows):
    """Abstand zur Schranke über das Budget: gewählter Modus gegen den jeweils anderen."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=DYNA_COLOR, width=2.5), name="Dynasearch"))
    fig.add_trace(go.Scatter(x=xs, y=[r["other"] for r in rows], mode="lines+markers", line=dict(color=SEQ_COLOR, width=2.5), name="Sequentiell"))
    fig.update_xaxes(title_text="Budget (bewertete Nachbarn)", type="log")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_sweep(rows, param_label):
    """Abstand zur Schranke (Dynasearch, Streuung als Band) und Sequentiell über die Werte eines Reglers."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    upper = [r["gap"] + r["gap_sd"] for r in rows]
    lower = [max(0.0, r["gap"] - r["gap_sd"]) for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(84,162,75,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=DYNA_COLOR, width=2.5), name="Dynasearch"))
    fig.add_trace(go.Scatter(x=xs, y=[r["other"] for r in rows], mode="lines+markers", line=dict(color=SEQ_COLOR, width=2, dash="dot"), name="Sequentiell"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_spread(run, other, mode_label, other_label):
    fig = go.Figure()
    fig.add_trace(go.Box(y=run, name=mode_label, marker_color=DYNA_COLOR, boxmean=True))
    fig.add_trace(go.Box(y=other, name=other_label, marker_color=SEQ_COLOR, boxmean=True))
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 360)


def build_scaling(blocks):
    """`blocks`: [{"label": ..., "rows": [...]}, ...] (siehe dyna_evaluation.scaling_table)."""
    fig = go.Figure()
    colors = (DYNA_COLOR, SEQ_COLOR)
    for block, color in zip(blocks, colors):
        label, rows = block["label"], block["rows"]
        xs = [r["value"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=label))
    fig.update_xaxes(title_text="Stopps")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)
