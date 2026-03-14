"""Dash application for spatialxgene."""

from __future__ import annotations

import base64
import datetime
import io
import threading
from typing import Optional, Tuple

import colorcet as cc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, Patch, callback_context, dash_table, ALL
import dash_bootstrap_components as dbc

import datashader as ds
import datashader.transfer_functions as tf

# Viridis colormap (256 colors) — matches Plotly's viridis
Viridis256 = [
    '#440154', '#440256', '#450457', '#450559', '#46075a', '#46085c', '#460a5d', '#460b5e',
    '#470d60', '#470e61', '#471063', '#471164', '#471365', '#481467', '#481668', '#481769',
    '#48186a', '#481a6c', '#481b6d', '#481c6e', '#481d6f', '#481f70', '#482071', '#482173',
    '#482374', '#482475', '#482576', '#482677', '#482878', '#482979', '#472a7a', '#472c7a',
    '#472d7b', '#472e7c', '#472f7d', '#46307e', '#46327e', '#46337f', '#463480', '#453581',
    '#453781', '#453882', '#443983', '#443a83', '#443b84', '#433d84', '#433e85', '#423f85',
    '#424086', '#424186', '#414287', '#414487', '#404588', '#404688', '#3f4788', '#3f4889',
    '#3e4989', '#3e4a89', '#3e4c8a', '#3d4d8a', '#3d4e8a', '#3c4f8a', '#3c508b', '#3b518b',
    '#3b528b', '#3a538b', '#3a548c', '#39558c', '#39568c', '#38588c', '#38598c', '#375a8c',
    '#375b8d', '#365c8d', '#365d8d', '#355e8d', '#355f8d', '#34608d', '#34618d', '#33628d',
    '#33638d', '#32648e', '#32658e', '#31668e', '#31678e', '#31688e', '#30698e', '#306a8e',
    '#2f6b8e', '#2f6c8e', '#2e6d8e', '#2e6e8e', '#2e6f8e', '#2d708e', '#2d718e', '#2c718e',
    '#2c728e', '#2c738e', '#2b748e', '#2b758e', '#2a768e', '#2a778e', '#2a788e', '#29798e',
    '#297a8e', '#297b8e', '#287c8e', '#287d8e', '#277e8e', '#277f8e', '#27808e', '#26818e',
    '#26828e', '#26828e', '#25838e', '#25848e', '#25858e', '#24868e', '#24878e', '#23888e',
    '#23898e', '#238a8d', '#228b8d', '#228c8d', '#228d8d', '#218e8d', '#218f8d', '#21908d',
    '#21918c', '#20928c', '#20928c', '#20938c', '#1f948c', '#1f958b', '#1f968b', '#1f978b',
    '#1f988b', '#1f998a', '#1f9a8a', '#1e9b8a', '#1e9c89', '#1e9d89', '#1f9e89', '#1f9f88',
    '#1fa088', '#1fa188', '#1fa187', '#20a287', '#20a386', '#21a486', '#21a585', '#22a685',
    '#22a785', '#23a884', '#24a983', '#25aa83', '#25ab82', '#26ac82', '#27ad81', '#28ae80',
    '#29af80', '#2ab07f', '#2cb17e', '#2db27d', '#2eb37c', '#2fb47c', '#31b57b', '#32b67a',
    '#34b679', '#35b778', '#37b877', '#38b976', '#3aba75', '#3cbb74', '#3dbc73', '#3fbd72',
    '#41be71', '#43bf6f', '#45c06e', '#47c16d', '#49c16c', '#4bc26b', '#4dc369', '#4fc468',
    '#51c567', '#54c665', '#56c764', '#58c762', '#5bc861', '#5dc960', '#60ca5e', '#62cb5c',
    '#65cb5b', '#67cc59', '#6acd58', '#6dcd56', '#6fce55', '#72cf53', '#75d051', '#77d050',
    '#7ad14e', '#7dd24c', '#80d24b', '#83d349', '#86d447', '#88d546', '#8bd544', '#8ed642',
    '#91d741', '#94d73f', '#97d83e', '#9ad83c', '#9dd93a', '#a0da39', '#a3da37', '#a6db36',
    '#a9dc34', '#acdc33', '#afdd31', '#b2dd30', '#b5de2f', '#b8de2e', '#bbdf2d', '#bedf2c',
    '#c1e02b', '#c4e02a', '#c7e12a', '#cae129', '#cde228', '#d0e228', '#d3e328', '#d6e327',
    '#d9e427', '#dce427', '#dfe527', '#e2e527', '#e5e628', '#e8e628', '#ebe729', '#eee72a',
    '#f1e82b', '#f4e82c', '#f7e92d', '#fae92f', '#fdea30',
]

# Datashader colormaps: all 256-color hex lists
COLORMAPS_DS = {
    'viridis':  Viridis256,
    'fire':     cc.fire,    # black → red → yellow → white
    'bmy':      cc.bmy,     # blue → magenta → yellow
    'bgy':      cc.bgy,     # blue → green → yellow
    'rainbow':  cc.rainbow, # full rainbow spectrum
    'dimgray':  cc.dimgray, # dark gray → light gray
}

def _make_plotly_colorscale(cmap: list) -> list:
    """Convert a 256-hex list to Plotly [[frac, color], ...] format."""
    n = len(cmap)
    return [[i / (n - 1), c] for i, c in enumerate(cmap)]

from .data import SpatialData

# sidebar width (keep in sync with CSS)
_SIDEBAR_W = 260

# Datashader canvas resolution (pixels) — higher = sharper points
_DS_WIDTH = 1600
_DS_HEIGHT = 1600

# Hover tooltips are expensive for large datasets (~90 bytes × N cells sent to browser).
# Auto-enable below this threshold; above it the user can toggle the checkbox manually.
_HOVER_AUTO_ON_MAX = 100_000

# ── Module-level DGE progress state (single-user local tool) ─────────────────

_dge_state: dict = {
    'running': False,
    'progress': 0,
    'label': '',
    'result': None,    # (df, n1, n2) once complete
    'error': None,     # error string if failed
}
_dge_lock = threading.Lock()


def _run_dge_thread(
    data_obj: SpatialData, g1_arr: np.ndarray, g2_arr: np.ndarray,
    test: str = 'ttest',
) -> None:
    """Background thread: runs DGE, updating _dge_state so the UI can poll progress."""
    try:
        needs_load = data_obj._csr_matrix is None and data_obj._csc_matrix is None
        with _dge_lock:
            _dge_state.update({
                'progress': 10,
                'label': ('Loading expression matrix — first time only, ~15 s…'
                          if needs_load else 'Slicing expression data…'),
            })

        data_obj._ensure_matrices()

        n1, n2 = len(g1_arr), len(g2_arr)
        ng = len(data_obj.gene_names)
        test_label = 'Wilcoxon' if test == 'wilcoxon' else 't-test'
        with _dge_lock:
            _dge_state.update({
                'progress': 55,
                'label': f'Running {test_label}  ({ng} genes × {n1 + n2:,} cells)…',
            })

        df = data_obj.run_dge(g1_arr, g2_arr, test=test)

        with _dge_lock:
            _dge_state.update({
                'running': False, 'progress': 100,
                'label': 'Complete', 'result': (df, n1, n2),
            })
    except Exception as exc:
        with _dge_lock:
            _dge_state.update({
                'running': False, 'progress': 0,
                'label': '', 'error': str(exc),
            })


# ── Colour resolution helper ──────────────────────────────────────────────────

def _resolve_color(
    data_obj: SpatialData,
    color_src: str,
    color_col: Optional[str],
    gene_col: Optional[str],
) -> Tuple[Optional[np.ndarray], bool, Optional[list], Optional[float], Optional[float]]:
    """Return (color_vals, is_categorical, cat_colors, vmin, vmax)."""
    if color_src == 'gene' and gene_col:
        expr = data_obj.get_gene_expr(gene_col)
        if expr is not None:
            vmin = float(np.nanmin(expr))
            vmax = float(np.nanmax(expr))
            return expr, False, None, vmin, vmax

    elif color_src == 'meta' and color_col:
        vals, is_cat, cat_colors = data_obj.get_color_info(color_col)
        if not is_cat and vals is not None:
            arr = np.asarray(vals, dtype=float)
            vmin = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0.0
            vmax = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1.0
        else:
            vmin = vmax = None
        return vals, is_cat, cat_colors, vmin, vmax

    return None, False, None, None, None


# ── Datashader rendering ───────────────────────────────────────────────────────

# Single-entry render cache: stores the last result to avoid re-rendering when
# only opacity changes (opacity is applied via layout_image, not baked into PNG).
_ds_cache: dict = {'key': None, 'result': None}

def _render_datashader(
    x: np.ndarray,
    y: np.ndarray,
    color_vals: Optional[np.ndarray],
    is_categorical: bool,
    cat_colors: Optional[list],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    width: int = _DS_WIDTH,
    height: int = _DS_HEIGHT,
    spread_px: int = 0,
    cmap_name: str = 'viridis',
) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
    """
    Render points with Datashader and return base64 PNG (transparent bg) + actual bounds.

    Returns: (base64_png, (x0, x1), (y0, y1))
    """
    # Compute ranges first so the cache key always uses explicit bounds
    if x_range is None:
        x_min, x_max = float(x.min()), float(x.max())
        x_pad = (x_max - x_min) * 0.02
        x_range = (x_min - x_pad, x_max + x_pad)
    if y_range is None:
        y_min, y_max = float(y.min()), float(y.max())
        y_pad = (y_max - y_min) * 0.02
        y_range = (y_min - y_pad, y_max + y_pad)

    # Build a hashable cache key from the render parameters
    _color_hash = None
    if color_vals is not None:
        _ca = np.asarray(color_vals)
        if _ca.dtype.kind in ('f', 'i', 'u'):
            _color_hash = (float(np.nanmean(_ca)), float(np.nanstd(_ca)), len(_ca))
        else:
            _color_hash = (str(_ca[:3]), len(_ca))
    _cat_hash = tuple((c, clr) for c, clr in cat_colors) if cat_colors else None
    cache_key = (id(x), len(x), x_range, y_range, is_categorical, _cat_hash,
                 _color_hash, spread_px, cmap_name, width, height)
    if _ds_cache['key'] == cache_key and _ds_cache['result'] is not None:
        return _ds_cache['result']

    df = pd.DataFrame({'x': x, 'y': y})

    canvas = ds.Canvas(plot_width=width, plot_height=height,
                       x_range=x_range, y_range=y_range)

    cmap = COLORMAPS_DS.get(cmap_name, Viridis256)

    if is_categorical and cat_colors is not None and color_vals is not None:
        df['cat'] = pd.Categorical(color_vals)
        agg = canvas.points(df, 'x', 'y', agg=ds.count_cat('cat'))
        color_key = {cat: color for cat, color in cat_colors}
        img = tf.shade(agg, color_key=color_key, how='eq_hist', min_alpha=220)

    elif color_vals is not None:
        df['val'] = color_vals
        agg = canvas.points(df, 'x', 'y', agg=ds.mean('val'))
        img = tf.shade(agg, cmap=cmap, how='linear', min_alpha=220)

    else:
        agg = canvas.points(df, 'x', 'y')
        img = tf.shade(agg, cmap=cmap, how='eq_hist', min_alpha=220)

    if spread_px > 0:
        img = tf.spread(img, px=spread_px, shape='circle')

    pil_img = img.to_pil()
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')

    result = (f'data:image/png;base64,{b64}', x_range, y_range)
    _ds_cache['key'] = cache_key
    _ds_cache['result'] = result
    return result


def _make_datashader_figure(
    x: np.ndarray,
    y: np.ndarray,
    color_vals: Optional[np.ndarray],
    is_categorical: bool,
    cat_colors: Optional[list],
    xlabel: str,
    ylabel: str,
    title: str,
    view: str,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    spread_px: int = 0,
    opacity: float = 1.0,
    cmap_name: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    hovertext: Optional[list] = None,
) -> go.Figure:
    """Build figure with Datashader raster + invisible selection layer + optional colorbar."""
    bg = '#1e1e1e'
    text = '#d4d4d4'
    grid = '#2a2a2a'

    img_b64, x_range, y_range = _render_datashader(
        x, y, color_vals, is_categorical, cat_colors, x_range, y_range,
        spread_px=spread_px, cmap_name=cmap_name,
    )

    hover_kwargs = {}
    if hovertext is not None:
        hover_kwargs = dict(
            hovertext=hovertext,
            hovertemplate='%{hovertext}<extra></extra>',
        )
        hoverinfo_val = 'text'
    else:
        hoverinfo_val = 'skip'

    selection_trace = go.Scattergl(
        x=x, y=y, mode='markers',
        marker=dict(size=3, opacity=0, color='white'),
        hoverinfo=hoverinfo_val,
        showlegend=False,
        **hover_kwargs,
    )

    traces = [selection_trace]

    # Colorbar trace for continuous data (invisible point, just renders the scale)
    if not is_categorical and color_vals is not None and vmin is not None and vmax is not None:
        cmap = COLORMAPS_DS.get(cmap_name, Viridis256)
        plotly_cs = _make_plotly_colorscale(cmap)
        colorbar_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                color=[vmin],
                colorscale=plotly_cs,
                showscale=True,
                cmin=vmin,
                cmax=vmax,
                opacity=0,
                colorbar=dict(
                    thickness=14,
                    len=0.5,
                    x=1.01,
                    xanchor='left',
                    tickfont=dict(color='#888', size=9),
                    title=dict(text='', font=dict(color='#888', size=9), side='right'),
                    bgcolor='rgba(0,0,0,0)',
                    outlinewidth=1,
                    outlinecolor='#3a3a3a',
                    borderwidth=0,
                    tickcolor='#555',
                ),
            ),
            hoverinfo='skip',
            showlegend=False,
        )
        traces.append(colorbar_trace)

    fig = go.Figure(data=traces)

    fig.add_layout_image(
        dict(
            source=img_b64,
            xref='x', yref='y',
            x=x_range[0], y=y_range[1],
            sizex=x_range[1] - x_range[0],
            sizey=y_range[1] - y_range[0],
            sizing='stretch',
            opacity=opacity,
            layer='below',
        )
    )

    yaxis_extra = dict(scaleanchor='x') if view == 'spatial' else {}

    fig.update_layout(
        title=dict(text=title, font=dict(color='#888', size=12), x=0, xref='paper',
                   xanchor='left', pad=dict(l=_SIDEBAR_W + 10)),
        paper_bgcolor=bg, plot_bgcolor=bg,
        xaxis=dict(
            title=xlabel, color=text, gridcolor=grid, zeroline=False, showspikes=False,
            range=list(x_range),
        ),
        yaxis=dict(
            title=ylabel, color=text, gridcolor=grid, zeroline=False, showspikes=False,
            range=list(y_range),
            **yaxis_extra,
        ),
        showlegend=False,
        margin=dict(l=_SIDEBAR_W + 10, r=60, t=36, b=50),
        uirevision=view,
        hovermode='closest' if hovertext is not None else False,
        autosize=True,
        dragmode='pan',
        modebar=dict(bgcolor='rgba(0,0,0,0)', color='#555', activecolor='#aaa',
                     orientation='v'),
    )

    return fig


# ── sidebar legend ────────────────────────────────────────────────────────────

def _sidebar_legend(cat_colors, obs_vals, active_cat=None):
    val_arr = np.asarray(obs_vals, dtype=object)
    items = []

    # "Show All" button when a category is isolated
    if active_cat is not None:
        items.append(html.Div(
            html.Span('Show All ✕', style={'fontSize': '10px', 'color': '#4ec9b0',
                                            'cursor': 'pointer'}),
            id={'type': 'legend-item', 'index': '__all__'},
            n_clicks=0,
            style={'marginBottom': '5px', 'cursor': 'pointer'},
        ))

    for cat, color in cat_colors:
        n = int((val_arr == cat).sum())
        is_active = active_cat is None or active_cat == str(cat)
        dim = 0.25 if (active_cat is not None and active_cat != str(cat)) else 1.0
        items.append(html.Div([
            html.Span(style={
                'display': 'inline-block', 'width': '10px', 'height': '10px',
                'borderRadius': '50%', 'background': color,
                'marginRight': '7px', 'flexShrink': '0',
                'opacity': str(dim),
            }),
            html.Span(f'{cat}  ({n:,})',
                      style={'fontSize': '11px', 'lineHeight': '1.4',
                             'color': '#bbb' if is_active else '#555'}),
        ], id={'type': 'legend-item', 'index': str(cat)}, n_clicks=0,
           style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '3px',
                  'cursor': 'pointer'}))
    return html.Div(items, style={'maxHeight': '260px', 'overflowY': 'auto'})


# ── DGE results table ─────────────────────────────────────────────────────────

def _dge_table(df: pd.DataFrame, n1: int, n2: int) -> html.Div:
    """Build a dark-themed DataTable from a DGE result DataFrame."""
    display = df.sort_values('log2fc', ascending=False).copy()
    display.insert(0, '#', range(1, len(display) + 1))
    display['log2fc'] = display['log2fc'].round(3)
    display['mean1']  = display['mean1'].round(4)
    display['mean2']  = display['mean2'].round(4)
    display['pval']   = display['pval'].map(lambda v: f'{v:.2e}')
    display['padj']   = display['padj'].map(lambda v: f'{v:.2e}')

    tbl = dash_table.DataTable(
        data=display.to_dict('records'),
        columns=[
            {'name': '#',       'id': '#',      'type': 'numeric'},
            {'name': 'Gene',    'id': 'gene'},
            {'name': 'Log₂FC',  'id': 'log2fc', 'type': 'numeric'},
            {'name': 'Mean G1', 'id': 'mean1',  'type': 'numeric'},
            {'name': 'Mean G2', 'id': 'mean2',  'type': 'numeric'},
            {'name': 'p-value', 'id': 'pval'},
            {'name': 'adj. p',  'id': 'padj'},
        ],
        sort_action='native',
        page_size=20,
        style_table={'overflowY': 'auto', 'overflowX': 'auto'},
        style_header={
            'backgroundColor': '#252526',
            'color': '#d4d4d4',
            'fontWeight': '600',
            'border': '1px solid #333',
            'fontSize': '11px',
            'textAlign': 'left',
        },
        style_cell={
            'backgroundColor': '#1e1e1e',
            'color': '#d4d4d4',
            'border': '1px solid #2a2a2a',
            'fontSize': '11px',
            'padding': '5px 10px',
            'textAlign': 'left',
        },
        style_data_conditional=[
            {'if': {'filter_query': '{log2fc} > 0', 'column_id': 'log2fc'},
             'color': '#4ec9b0', 'fontWeight': '600'},
            {'if': {'filter_query': '{log2fc} < 0', 'column_id': 'log2fc'},
             'color': '#f48771', 'fontWeight': '600'},
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#222222'},
        ],
    )

    subtitle = html.Div(
        f'Top {len(display)} of {len(df)} genes  ·  '
        f'Group 1: {n1:,} cells  ·  Group 2: {n2:,} cells',
        style={'color': '#777', 'fontSize': '11px', 'marginBottom': '8px'},
    )
    legend = html.Div([
        html.Span('● ', style={'color': '#4ec9b0'}),
        html.Span('up in G1   ', style={'color': '#aaa', 'fontSize': '11px',
                                        'marginRight': '12px'}),
        html.Span('● ', style={'color': '#f48771'}),
        html.Span('up in G2', style={'color': '#aaa', 'fontSize': '11px'}),
    ], style={'marginBottom': '10px'})

    return html.Div([subtitle, legend, tbl])


# ── app factory ───────────────────────────────────────────────────────────────

def create_app(data: SpatialData) -> dash.Dash:
    views   = data.available_views()
    columns = data.color_columns()
    genes   = data.gene_names
    has_dge = len(genes) > 0

    if not views:
        raise ValueError(
            f"No 2-D embeddings found in '{data.path.name}'.\n"
            "spatialxgene needs at least one of:\n"
            "  • obsm keys: spatial, X_umap, X_pca, X_scVI (or any 2-D obsm array)\n"
            "  • obs columns: center_x/center_y, x/y, x_centroid/y_centroid, spatial_x/spatial_y"
        )
    default_view  = views[0]['value']
    default_color = next(
        (c['value'] for c in columns if c['is_categorical']),
        columns[0]['value'] if columns else None,
    )

    # ── label helper ──────────────────────────────────────────────────

    def _label(text):
        return html.Label(text, style={
            'color': '#666', 'fontSize': '10px',
            'textTransform': 'uppercase', 'letterSpacing': '1px',
            'display': 'block', 'marginBottom': '5px', 'marginTop': '12px',
        })

    gene_opts = [{'label': g, 'value': g} for g in sorted(genes)]

    # ── DGE sidebar panel ─────────────────────────────────────────────

    dge_panel = html.Div([
        html.Hr(style={'borderColor': '#2d2d2d', 'margin': '12px 0 8px'}),
        html.Div('DGE Analysis', style={
            'color': '#666', 'fontSize': '10px',
            'textTransform': 'uppercase', 'letterSpacing': '1px',
            'marginBottom': '5px',
        }),
        html.Div(
            'Lasso  or box-select ⬜ cells on the plot, then assign to a group.',
            style={'color': '#555', 'fontSize': '10px', 'lineHeight': '1.5',
                   'marginBottom': '10px'},
        ),

        # Group 1
        html.Div([
            html.Span('●', style={'color': '#4ec9b0', 'marginRight': '5px',
                                   'fontSize': '12px', 'lineHeight': '1'}),
            html.Span('Group 1', style={'color': '#9a9a9a', 'fontSize': '11px',
                                        'marginRight': 'auto'}),
            html.Span(id='dge-g1-label', children='not set',
                      style={'color': '#666', 'fontSize': '10px', 'marginRight': '6px'}),
            html.Button('Set', id='dge-set-g1', n_clicks=0, className='dge-set-btn'),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '7px'}),

        # Group 2
        html.Div([
            html.Span('●', style={'color': '#f48771', 'marginRight': '5px',
                                   'fontSize': '12px', 'lineHeight': '1'}),
            html.Span('Group 2', style={'color': '#9a9a9a', 'fontSize': '11px',
                                        'marginRight': 'auto'}),
            html.Span(id='dge-g2-label', children='not set',
                      style={'color': '#666', 'fontSize': '10px', 'marginRight': '6px'}),
            html.Button('Set', id='dge-set-g2', n_clicks=0, className='dge-set-btn'),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),

        # G2 = rest toggle
        html.Div(
            dcc.Checklist(
                id='g2-rest-check',
                options=[{'label': ' G2 = all other cells', 'value': 'g2_rest'}],
                value=[],
                inputStyle={'marginRight': '4px'},
                labelStyle={'color': '#888', 'fontSize': '10px',
                            'display': 'inline-block'},
            ),
            style={'marginBottom': '8px'},
        ),

        # Test type
        _label('Test'),
        dcc.RadioItems(
            id='dge-test-radio',
            options=[
                {'label': ' t-test', 'value': 'ttest'},
                {'label': ' Wilcoxon', 'value': 'wilcoxon'},
            ],
            value='ttest',
            labelStyle={'display': 'inline-block', 'marginRight': '10px',
                        'color': '#aaa', 'fontSize': '11px'},
            inputStyle={'marginRight': '4px'},
            style={'marginBottom': '8px'},
        ),

        # Buttons
        html.Div([
            html.Button('Clear', id='dge-clear-btn', n_clicks=0,
                        className='dge-action-btn'),
            html.Button('⬇ G1', id='dl-g1-btn', n_clicks=0,
                        className='dge-action-btn', title='Download G1 barcodes'),
            html.Button('⬇ G2', id='dl-g2-btn', n_clicks=0,
                        className='dge-action-btn', title='Download G2 barcodes'),
            html.Button('Run DGE ▶', id='dge-run-btn', n_clicks=0,
                        disabled=True, className='dge-run-btn'),
        ], style={'display': 'flex', 'gap': '6px'}),

        # History section
        html.Div(id='dge-history-display', style={'marginTop': '10px'}),

    ], style={'display': 'block'} if has_dge else {'display': 'none'})

    # ── full sidebar ──────────────────────────────────────────────────

    sidebar = html.Div([
        # Logo
        html.Div([
            html.Span('spatial', style={'color': '#4ec9b0', 'fontWeight': '700',
                                        'fontSize': '18px'}),
            html.Span('xgene',   style={'color': '#9cdcfe', 'fontWeight': '700',
                                        'fontSize': '18px'}),
        ], style={'marginBottom': '14px'}),

        # --- Embedding ---
        _label('Embedding'),
        dcc.RadioItems(
            id='view-radio', options=views, value=default_view,
            labelStyle={'display': 'block', 'marginBottom': '3px',
                        'color': '#ccc', 'fontSize': '12px'},
            inputStyle={'marginRight': '6px'},
        ),

        # --- Color source ---
        _label('Color by'),
        dcc.RadioItems(
            id='color-source',
            options=[{'label': ' Metadata', 'value': 'meta'},
                     {'label': ' Gene expression', 'value': 'gene'}],
            value='meta',
            labelStyle={'display': 'inline-block', 'marginRight': '12px',
                        'color': '#ccc', 'fontSize': '12px'},
            inputStyle={'marginRight': '4px'},
            style={'marginBottom': '6px'},
        ),

        # metadata dropdown
        html.Div(
            dcc.Dropdown(
                id='color-dropdown',
                options=[{'label': c['label'], 'value': c['value']} for c in columns],
                value=default_color, clearable=False, searchable=True,
                style={'fontSize': '11px'},
                className='dark-dropdown',
            ),
            id='meta-div',
        ),

        # gene dropdown
        html.Div(
            dcc.Dropdown(
                id='gene-dropdown',
                options=gene_opts,
                value=None, clearable=True, searchable=True,
                placeholder='Search gene…',
                style={'fontSize': '11px'},
                className='dark-dropdown',
            ),
            id='gene-div',
            style={'display': 'none'},
        ),

        # Gene stats line (shown when a gene is selected)
        html.Div(id='gene-stats',
                 style={'color': '#666', 'fontSize': '10px', 'marginTop': '4px',
                        'fontStyle': 'italic'}),

        # --- Point size slider ---
        _label('Point size'),
        dcc.Slider(id='size-slider', min=0, max=6, step=1, value=2,
                   marks={i: {'label': str(i), 'style': {'color': '#ccc'}}
                          for i in [0, 2, 4, 6]},
                   tooltip={'placement': 'bottom'}, className='dark-slider'),

        # --- Opacity slider ---
        _label('Opacity'),
        dcc.Slider(id='opacity-slider', min=0.1, max=1.0, step=0.1, value=1.0,
                   marks={0.2: {'label': '.2', 'style': {'color': '#ccc'}},
                          0.5: {'label': '.5', 'style': {'color': '#ccc'}},
                          1.0: {'label': '1', 'style': {'color': '#ccc'}}},
                   tooltip={'placement': 'bottom'}, className='dark-slider'),

        # --- Colormap (applies to continuous data) ---
        _label('Colormap'),
        dcc.RadioItems(
            id='colormap-radio',
            options=[
                {'label': ' Viridis',  'value': 'viridis'},
                {'label': ' Fire',     'value': 'fire'},
                {'label': ' BMY',      'value': 'bmy'},
                {'label': ' BGY',      'value': 'bgy'},
                {'label': ' Rainbow',  'value': 'rainbow'},
                {'label': ' Gray',     'value': 'dimgray'},
            ],
            value='viridis',
            labelStyle={'display': 'inline-block', 'marginRight': '8px',
                        'color': '#888', 'fontSize': '11px'},
            inputStyle={'marginRight': '4px'},
            style={'marginBottom': '4px'},
        ),

        # --- Color range clipping (continuous data only) ---
        html.Div([
            _label('Color range'),
            dcc.RangeSlider(
                id='color-range-slider',
                min=0, max=1, step=0.01,
                value=[0, 1],
                marks={},
                tooltip={'placement': 'bottom'},
                className='dark-slider',
            ),
            html.Div(id='color-range-label',
                     style={'color': '#666', 'fontSize': '10px',
                            'textAlign': 'center', 'marginTop': '2px'}),
        ], id='color-range-div', style={'display': 'none'}),

        # --- Axis flips ---
        html.Div(
            dcc.Checklist(
                id='flip-check',
                options=[
                    {'label': ' Flip X', 'value': 'flip_x'},
                    {'label': ' Flip Y', 'value': 'flip_y'},
                ],
                value=['flip_y'],
                inputStyle={'marginRight': '4px'},
                labelStyle={'color': '#ccc', 'fontSize': '12px',
                            'display': 'inline-block', 'marginRight': '10px'},
            ),
            style={'marginTop': '10px'},
        ),

        # --- Hover tooltips toggle ---
        html.Div(
            dcc.Checklist(
                id='hover-check',
                options=[{'label': ' Hover tooltips', 'value': 'show_hover'}],
                # Auto-on for small datasets; large datasets default off (performance)
                value=['show_hover'] if data.n_cells <= _HOVER_AUTO_ON_MAX else [],
                inputStyle={'marginRight': '4px'},
                labelStyle={'color': '#ccc', 'fontSize': '12px',
                            'display': 'inline-block'},
            ),
            style={'marginTop': '6px'},
        ),

        # --- DGE panel ---
        dge_panel,

        # --- Legend ---
        html.Hr(style={'borderColor': '#333', 'margin': '12px 0 8px'}),
        html.Div(id='legend-div'),

    ], className='spatialxgene-sidebar')

    # ── DGE modal ─────────────────────────────────────────────────────

    dge_modal = dbc.Modal([
        dbc.ModalHeader(
            html.Span('DGE Results', style={'color': '#4ec9b0', 'fontWeight': '700'}),
            close_button=False,
            style={'background': '#252526', 'borderBottom': '1px solid #333'},
        ),
        dbc.ModalBody([
            html.Div([
                html.Div(id='dge-progress-label',
                         style={'color': '#aaa', 'fontSize': '12px',
                                'marginBottom': '8px', 'fontStyle': 'italic'}),
                dbc.Progress(
                    id='dge-progress-bar', value=0,
                    striped=True, animated=True,
                    className='dge-progress',
                    style={'height': '8px'},
                ),
            ], id='dge-progress-section',
               style={'display': 'none', 'marginBottom': '20px'}),

            html.Div(id='dge-results'),

        ], style={'background': '#1e1e1e', 'padding': '16px'}),
        dbc.ModalFooter([
            dbc.Button(
                '⬇ Download CSV', id='dge-download-btn', color='secondary', size='sm',
                disabled=True,
                style={'marginRight': 'auto', 'fontSize': '11px'},
            ),
            dbc.Button('Close', id='dge-modal-close', color='secondary', size='sm'),
        ], style={'background': '#252526', 'borderTop': '1px solid #333'}),
    ], id='dge-modal', is_open=False, size='xl', centered=True,
       backdrop='static', scrollable=True)

    # ── Dash app ──────────────────────────────────────────────────────

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title='spatialxgene',
        assets_folder=str(__file__).replace('app.py', 'assets'),
        suppress_callback_exceptions=True,
    )

    # Inline the loading-spinner CSS in the index so it appears BEFORE
    # external stylesheets and JS bundles finish loading.
    app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <style>
            body { background: #1e1e1e; margin: 0; }
            ._dash-loading {
                display: flex !important;
                align-items: center;
                justify-content: center;
                height: 100vh;
                background: #1e1e1e;
            }
            ._dash-loading::after {
                content: '';
                width: 36px;
                height: 36px;
                border: 3px solid #333;
                border-top-color: #4ec9b0;
                border-radius: 50%;
                animation: sxg-spin 0.8s linear infinite;
            }
            @keyframes sxg-spin { to { transform: rotate(360deg); } }
        </style>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

    app.layout = html.Div([
        # Stores
        dcc.Store(id='dge-current-sel', data=[]),
        dcc.Store(id='dge-g1',          data=[]),
        dcc.Store(id='dge-g2',          data=[]),
        dcc.Store(id='current-view-data', data={}),
        dcc.Store(id='zoom-state', data={}),
        dcc.Store(id='dge-history', data=[]),
        dcc.Store(id='dge-result-store', data=None),  # serialised DGE df for download
        dcc.Store(id='legend-filter', data=None),       # active category for legend isolate

        # Download targets
        dcc.Download(id='dge-download'),
        dcc.Download(id='barcode-download'),

        # Interval for DGE progress polling
        dcc.Interval(id='dge-interval', interval=500, n_intervals=0, disabled=True),

        # Full-screen graph
        dcc.Graph(
            id='main-scatter',
            config={
                'scrollZoom': True,
                'toImageButtonOptions': {'format': 'svg', 'filename': 'spatialxgene'},
                'displaylogo': False,
                'modeBarButtonsToRemove': [],
                'doubleClick': 'reset',
            },
            className='full-screen-graph',
        ),

        sidebar,
        html.Div(id='cell-count', className='cell-count-badge'),
        dge_modal,

    ], style={'background': '#1e1e1e'})

    # ── callbacks ─────────────────────────────────────────────────────

    @app.callback(
        Output('meta-div', 'style'),
        Output('gene-div', 'style'),
        Input('color-source', 'value'),
    )
    def toggle_color_source(src):
        if src == 'gene':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'block'}, {'display': 'none'}

    @app.callback(
        Output('gene-stats', 'children'),
        Input('gene-dropdown', 'value'),
        Input('color-source',  'value'),
    )
    def update_gene_stats(gene, color_src):
        if color_src != 'gene' or not gene:
            return ''
        expr = data.get_gene_expr(gene)
        if expr is None:
            return ''
        total = len(expr)
        n_expr = int(np.count_nonzero(expr))
        pct = 100.0 * n_expr / total if total else 0
        return f'{pct:.1f}% expressing  ·  mean {np.mean(expr):.2f}  ·  max {np.max(expr):.1f}'

    @app.callback(
        Output('flip-check', 'value'),
        Input('view-radio', 'value'),
    )
    def reset_flips_on_view_change(view):
        return ['flip_y'] if view == 'spatial' else []

    # Legend click → isolate category
    @app.callback(
        Output('legend-filter', 'data'),
        Input({'type': 'legend-item', 'index': ALL}, 'n_clicks'),
        State('legend-filter', 'data'),
        prevent_initial_call=True,
    )
    def legend_click(n_clicks_list, current_filter):
        if not callback_context.triggered:
            return dash.no_update
        triggered = callback_context.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return dash.no_update
        clicked = triggered.get('index')
        if clicked == '__all__' or clicked == current_filter:
            return None  # clear filter
        return clicked

    # Clear legend filter when color column changes
    @app.callback(
        Output('legend-filter', 'data', allow_duplicate=True),
        Input('color-source',  'value'),
        Input('color-dropdown', 'value'),
        Input('gene-dropdown',  'value'),
        prevent_initial_call=True,
    )
    def clear_legend_filter_on_color_change(src, col, gene):
        return None

    # Update color range slider bounds when color source/column/gene changes
    @app.callback(
        Output('color-range-div',    'style'),
        Output('color-range-slider', 'min'),
        Output('color-range-slider', 'max'),
        Output('color-range-slider', 'step'),
        Output('color-range-slider', 'value'),
        Output('color-range-slider', 'marks'),
        Output('color-range-label',  'children'),
        Input('color-source',  'value'),
        Input('color-dropdown', 'value'),
        Input('gene-dropdown',  'value'),
    )
    def update_color_range(color_src, color_col, gene_col):
        _, is_cat, _, vmin, vmax = _resolve_color(data, color_src, color_col, gene_col)
        if is_cat or vmin is None or vmax is None or vmin == vmax:
            return {'display': 'none'}, 0, 1, 0.01, [0, 1], {}, ''
        step = (vmax - vmin) / 200
        marks = {
            vmin: {'label': f'{vmin:.2g}', 'style': {'color': '#888', 'fontSize': '9px'}},
            vmax: {'label': f'{vmax:.2g}', 'style': {'color': '#888', 'fontSize': '9px'}},
        }
        label = f'{vmin:.2g} – {vmax:.2g}'
        return {'display': 'block'}, vmin, vmax, step, [vmin, vmax], marks, label

    # Main figure callback
    @app.callback(
        Output('main-scatter',    'figure'),
        Output('legend-div',      'children'),
        Output('cell-count',      'children'),
        Output('current-view-data', 'data'),
        Output('zoom-state',      'data'),
        Input('view-radio',    'value'),
        Input('flip-check',    'value'),
        Input('color-source',  'value'),
        Input('color-dropdown','value'),
        Input('gene-dropdown', 'value'),
        Input('size-slider',   'value'),
        Input('opacity-slider','value'),
        Input('colormap-radio','value'),
        Input('hover-check',   'value'),
        Input('color-range-slider', 'value'),
        Input('legend-filter', 'data'),
        State('zoom-state',    'data'),
    )
    def update_figure(view, flip_vals, color_src, color_col, gene_col,
                      spread_px, opacity, cmap_name, hover_vals, color_range,
                      legend_filter, zoom_state):
        triggered = callback_context.triggered_id
        x_range = None
        y_range = None
        visual_only = {'size-slider', 'opacity-slider', 'colormap-radio', 'hover-check',
                       'color-range-slider', 'legend-filter'}
        if triggered in visual_only and zoom_state:
            xr = zoom_state.get('x_range')
            yr = zoom_state.get('y_range')
            if xr:
                x_range = tuple(xr)
            if yr:
                y_range = tuple(yr)

        flip_y = 'flip_y' in (flip_vals or [])
        flip_x = 'flip_x' in (flip_vals or [])
        x, y = data.get_coords(view, flip_y=flip_y, flip_x=flip_x)

        if x is None:
            empty = go.Figure()
            empty.update_layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
                                margin=dict(l=_SIDEBAR_W + 10, r=60, t=36, b=50),
                                autosize=True)
            return empty, html.Div(), '', {}, {}

        xlabel, ylabel = data.axis_labels(view)
        if flip_y:
            ylabel += ' (flipped)'
        if flip_x:
            xlabel += ' (flipped)'

        cmap_name = cmap_name or 'viridis'
        color_vals, is_categorical, cat_colors, vmin, vmax = _resolve_color(
            data, color_src, color_col, gene_col
        )

        # Apply user color range clipping for continuous data
        if not is_categorical and color_vals is not None and color_range:
            clip_lo, clip_hi = color_range
            if vmin is not None and vmax is not None and clip_hi > clip_lo:
                vmin, vmax = float(clip_lo), float(clip_hi)
                color_vals = np.clip(color_vals, vmin, vmax)

        legend_content = html.Div()
        if is_categorical and cat_colors:
            legend_content = _sidebar_legend(cat_colors, color_vals, active_cat=legend_filter)
            # When a category is isolated, grey out all others in the datashader render
            if legend_filter is not None:
                filtered_colors = [(cat, color if str(cat) == legend_filter else '#222222')
                                   for cat, color in cat_colors]
                cat_colors = filtered_colors
            label = f'{data.name}  ·  {view.upper()}  ·  {color_col}'
        elif color_src == 'gene' and gene_col:
            label = f'{data.name}  ·  {view.upper()}  ·  {gene_col}'
        elif color_src == 'meta' and color_col:
            label = f'{data.name}  ·  {view.upper()}  ·  {color_col}'
        else:
            label = f'{data.name}  ·  {view.upper()}'

        # Hover text: only built when the toggle is on (large datasets are expensive)
        show_hover = 'show_hover' in (hover_vals or [])
        hovertext = None
        if show_hover and color_vals is not None:
            col_label = gene_col if color_src == 'gene' else color_col
            barcodes = list(data.obs.index)
            if is_categorical:
                obs_vals = data.obs[color_col]
                hovertext = [
                    f'<b>{col_label}</b>: {v}<br>'
                    f'<span style="color:#888;font-size:10px">{b}</span>'
                    for v, b in zip(obs_vals, barcodes)
                ]
            else:
                hovertext = [
                    f'<b>{col_label}</b>: {v:.3g}<br>'
                    f'<span style="color:#888;font-size:10px">{b}</span>'
                    for v, b in zip(color_vals, barcodes)
                ]

        fig = _make_datashader_figure(
            x, y, color_vals, is_categorical, cat_colors,
            xlabel, ylabel, label, view,
            x_range=x_range, y_range=y_range,
            spread_px=spread_px or 0,
            opacity=opacity or 1.0,
            cmap_name=cmap_name,
            vmin=vmin, vmax=vmax,
            hovertext=hovertext,
        )

        if data.n_cells < data.n_cells_total:
            count_str = f'{data.n_cells:,} shown · {data.n_cells_total:,} total'
        else:
            count_str = f'{data.n_cells:,} cells'

        view_data = {
            'view': view, 'flip_y': flip_y, 'flip_x': flip_x,
            'color_src': color_src, 'color_col': color_col, 'gene_col': gene_col,
            'xlabel': xlabel, 'ylabel': ylabel, 'label': label,
            'spread_px': spread_px or 0, 'opacity': opacity or 1.0,
            'cmap_name': cmap_name,
        }

        new_zoom = zoom_state if triggered in visual_only else {}
        return fig, legend_content, count_str, view_data, new_zoom

    # Re-render on zoom/pan using Patch — only updates the image, not the trace data
    @app.callback(
        Output('main-scatter', 'figure', allow_duplicate=True),
        Output('zoom-state',   'data', allow_duplicate=True),
        Input('main-scatter', 'relayoutData'),
        State('current-view-data', 'data'),
        prevent_initial_call=True,
    )
    def update_on_relayout(relayout_data, view_data):
        if not relayout_data or not view_data:
            return dash.no_update, dash.no_update

        x_range = None
        y_range = None
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            x_range = (relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]'])
        if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
            y_range = (relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]'])

        if x_range is None and y_range is None:
            return dash.no_update, dash.no_update
        if 'autosize' in relayout_data or 'xaxis.autorange' in relayout_data:
            return dash.no_update, dash.no_update

        view    = view_data.get('view', 'spatial')
        flip_y  = view_data.get('flip_y', False)
        flip_x  = view_data.get('flip_x', False)
        cmap_name = view_data.get('cmap_name', 'viridis')

        x, y = data.get_coords(view, flip_y=flip_y, flip_x=flip_x)
        if x is None:
            return dash.no_update, dash.no_update

        color_vals, is_categorical, cat_colors, _, _ = _resolve_color(
            data,
            view_data.get('color_src', 'meta'),
            view_data.get('color_col'),
            view_data.get('gene_col'),
        )

        img_b64, x_range, y_range = _render_datashader(
            x, y, color_vals, is_categorical, cat_colors,
            x_range=x_range, y_range=y_range,
            spread_px=view_data.get('spread_px', 0),
            cmap_name=cmap_name,
        )

        # Patch: only update the image — scatter trace data (incl. hover) stays intact
        patch = Patch()
        patch['layout']['images'][0]['source'] = img_b64
        patch['layout']['images'][0]['x']      = x_range[0]
        patch['layout']['images'][0]['y']      = y_range[1]
        patch['layout']['images'][0]['sizex']  = x_range[1] - x_range[0]
        patch['layout']['images'][0]['sizey']  = y_range[1] - y_range[0]
        patch['layout']['xaxis']['range']      = list(x_range)
        patch['layout']['yaxis']['range']      = list(y_range)

        new_zoom = {'x_range': list(x_range), 'y_range': list(y_range)}
        return patch, new_zoom

    # ── DGE selection callbacks ────────────────────────────────────────

    @app.callback(
        Output('dge-current-sel', 'data'),
        Input('main-scatter', 'selectedData'),
        prevent_initial_call=True,
    )
    def store_current_selection(selected_data):
        if not selected_data or not selected_data.get('points'):
            return []
        return [p['pointIndex'] for p in selected_data['points']]

    @app.callback(
        Output('dge-g1', 'data'),
        Input('dge-set-g1', 'n_clicks'),
        State('dge-current-sel', 'data'),
        prevent_initial_call=True,
    )
    def set_group1(_, current_sel):
        return current_sel or dash.no_update

    @app.callback(
        Output('dge-g2', 'data'),
        Input('dge-set-g2', 'n_clicks'),
        State('dge-current-sel', 'data'),
        prevent_initial_call=True,
    )
    def set_group2(_, current_sel):
        return current_sel or dash.no_update

    @app.callback(
        Output('dge-g1', 'data', allow_duplicate=True),
        Output('dge-g2', 'data', allow_duplicate=True),
        Output('dge-current-sel', 'data', allow_duplicate=True),
        Input('dge-clear-btn', 'n_clicks'),
        prevent_initial_call=True,
    )
    def clear_groups(_):
        return [], [], []

    @app.callback(
        Output('dge-g1-label', 'children'),
        Output('dge-g2-label', 'children'),
        Output('dge-run-btn',  'disabled'),
        Input('dge-g1', 'data'),
        Input('dge-g2', 'data'),
        Input('g2-rest-check', 'value'),
    )
    def update_dge_ui(g1, g2, g2_rest_vals):
        n1 = len(g1) if g1 else 0
        n2 = len(g2) if g2 else 0
        g2_is_rest = 'g2_rest' in (g2_rest_vals or [])
        if g2_is_rest and n1 > 0:
            n2_eff = data.n_cells - n1
            g2_label = f'{n2_eff:,} cells (rest)'
            can_run = n1 > 0
        else:
            g2_label = f'{n2:,} cells' if n2 else 'not set'
            can_run = n1 > 0 and n2 > 0
        return (
            f'{n1:,} cells' if n1 else 'not set',
            g2_label,
            not can_run,
        )

    # ── DGE run / progress / close ─────────────────────────────────────

    @app.callback(
        Output('dge-modal',    'is_open'),
        Output('dge-interval', 'disabled', allow_duplicate=True),
        Output('main-scatter', 'figure', allow_duplicate=True),
        Input('dge-run-btn',   'n_clicks'),
        State('dge-g1',        'data'),
        State('dge-g2',        'data'),
        State('dge-test-radio', 'value'),
        State('g2-rest-check', 'value'),
        prevent_initial_call=True,
    )
    def start_dge(_, g1, g2, test, g2_rest_vals):
        g2_is_rest = 'g2_rest' in (g2_rest_vals or [])
        if g2_is_rest and g1:
            g1_set = set(g1)
            g2 = [i for i in range(data.n_cells) if i not in g1_set]
        if not g1 or not g2:
            return False, True, dash.no_update

        with _dge_lock:
            if _dge_state['running']:
                return True, False, dash.no_update

        g1_arr = np.array(g1, dtype=int)
        g2_arr = np.array(g2, dtype=int)

        with _dge_lock:
            _dge_state.update({
                'running': True, 'progress': 0,
                'label': 'Starting…', 'result': None, 'error': None,
            })

        threading.Thread(
            target=_run_dge_thread,
            args=(data, g1_arr, g2_arr, test or 'ttest'),
            daemon=True,
        ).start()

        patch = Patch()
        patch['data'][0]['selectedpoints'] = []
        patch['layout']['selections'] = []
        return True, False, patch

    @app.callback(
        Output('dge-progress-bar',     'value'),
        Output('dge-progress-label',   'children'),
        Output('dge-progress-section', 'style'),
        Output('dge-results',          'children'),
        Output('dge-interval',         'disabled'),
        Output('dge-history',          'data'),
        Output('dge-result-store',     'data'),
        Output('dge-download-btn',     'disabled'),
        Input('dge-interval',          'n_intervals'),
        State('dge-history',           'data'),
        prevent_initial_call=True,
    )
    def poll_dge_progress(_, history):
        with _dge_lock:
            s = dict(_dge_state)

        show = {'display': 'block', 'marginBottom': '20px'}
        hide = {'display': 'none'}
        history = history or []

        if s['error']:
            return (
                0, f"Error: {s['error']}", show,
                html.Div(f"Error: {s['error']}",
                         style={'color': '#f48771', 'fontSize': '12px'}),
                True, history, dash.no_update, True,
            )

        if not s['running'] and s['result'] is not None:
            df, n1, n2 = s['result']

            df_fc = df.sort_values('log2fc', ascending=False)
            up_genes   = df_fc[df_fc['log2fc'] > 0].head(10)[['gene', 'log2fc', 'padj']].to_dict('records')
            down_genes = df_fc[df_fc['log2fc'] < 0].tail(10)[['gene', 'log2fc', 'padj']].to_dict('records')
            entry = {
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'n1': n1, 'n2': n2, 'n_genes': len(df),
                'up_genes': up_genes, 'down_genes': down_genes,
            }
            history = history + [entry]

            result_store = {
                'records': df.sort_values('log2fc', ascending=False).to_dict('records'),
                'n1': n1, 'n2': n2,
            }

            return (
                100, 'Complete', hide,
                _dge_table(df, n1, n2),
                True, history, result_store, False,
            )

        return (
            s['progress'], s['label'], show,
            html.Div(), False, history, dash.no_update, True,
        )

    # DGE CSV download
    @app.callback(
        Output('dge-download', 'data'),
        Input('dge-download-btn', 'n_clicks'),
        State('dge-result-store', 'data'),
        prevent_initial_call=True,
    )
    def download_dge_csv(_, result_data):
        if not result_data:
            return dash.no_update
        df = pd.DataFrame(result_data['records'])
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return dcc.send_data_frame(df.to_csv, f'dge_{timestamp}.csv', index=False)

    # Barcode download (G1 / G2)
    @app.callback(
        Output('barcode-download', 'data'),
        Input('dl-g1-btn', 'n_clicks'),
        Input('dl-g2-btn', 'n_clicks'),
        State('dge-g1', 'data'),
        State('dge-g2', 'data'),
        prevent_initial_call=True,
    )
    def download_barcodes(n1, n2, g1, g2):
        triggered = callback_context.triggered_id
        if triggered == 'dl-g1-btn' and g1:
            barcodes = [data.obs.index[i] for i in g1]
            label = 'G1'
        elif triggered == 'dl-g2-btn' and g2:
            barcodes = [data.obs.index[i] for i in g2]
            label = 'G2'
        else:
            return dash.no_update
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        content = '\n'.join(barcodes) + '\n'
        return dict(content=content, filename=f'barcodes_{label}_{timestamp}.txt')

    # DGE history display
    @app.callback(
        Output('dge-history-display', 'children'),
        Input('dge-history', 'data'),
    )
    def update_history_display(history):
        if not history:
            return html.Div()

        items = []
        for i, entry in enumerate(reversed(history)):
            run_num = len(history) - i
            up_genes   = entry.get('up_genes', [])
            down_genes = entry.get('down_genes', [])

            up_str   = ', '.join(g['gene'] for g in up_genes[:5])   if up_genes   else '—'
            down_str = ', '.join(g['gene'] for g in down_genes[:5]) if down_genes else '—'

            items.append(html.Div([
                html.Div([
                    html.Span(f'#{run_num}', style={
                        'color': '#888', 'fontSize': '11px', 'fontWeight': '600',
                        'marginRight': '6px',
                    }),
                    html.Span(f"{entry['timestamp']}", style={
                        'color': '#666', 'fontSize': '10px', 'marginRight': '6px',
                    }),
                    html.Span(f"G1:{entry['n1']:,}  G2:{entry['n2']:,}", style={
                        'color': '#555', 'fontSize': '10px',
                    }),
                ], style={'marginBottom': '2px'}),
                html.Div([
                    html.Span('▲ G1: ', style={'color': '#4ec9b0', 'fontSize': '11px'}),
                    html.Span(up_str,   style={'color': '#4ec9b0', 'fontSize': '11px'}),
                ], style={'marginBottom': '1px'}),
                html.Div([
                    html.Span('▼ G2: ',  style={'color': '#f48771', 'fontSize': '11px'}),
                    html.Span(down_str, style={'color': '#f48771', 'fontSize': '11px'}),
                ]),
            ], style={
                'marginBottom': '8px', 'paddingBottom': '6px',
                'borderBottom': '1px solid #2a2a2a',
            }))

        return html.Div([
            html.Div('History', style={
                'color': '#555', 'fontSize': '10px', 'textTransform': 'uppercase',
                'letterSpacing': '1px', 'marginBottom': '6px', 'marginTop': '8px',
            }),
            html.Div(items, style={'maxHeight': '160px', 'overflowY': 'auto'}),
        ])

    @app.callback(
        Output('dge-modal',    'is_open', allow_duplicate=True),
        Output('dge-interval', 'disabled', allow_duplicate=True),
        Input('dge-modal-close', 'n_clicks'),
        prevent_initial_call=True,
    )
    def close_dge_modal(_):
        return False, True

    return app
