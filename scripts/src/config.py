import os

import plotnine as gg

MERFISH_PANCANCER_DIR = os.environ["MERFISH_DIR"]
FIGURE_DIR = os.path.join(os.environ["REPO_DIR"], "figures")
RESULTS_DIR = os.path.join(os.environ["REPO_DIR"], "results")
MAIN_ADATA = "lung2_adata.annotated.h5ad"
MAIN_SAMPLE_NAME = "HumanLungCancerPatient2"


# FONT_FAMILY = "Arial"
FONT_FAMILY = "sans-serif"
DEFAULT_THEME = gg.theme(
    # Half-open frame: thin left + bottom axes only
    axis_line=gg.element_line(color="#333333", size=0.4),
    axis_line_x=gg.element_line(color="#333333", size=0.4),
    axis_line_y=gg.element_line(color="#333333", size=0.4),
    # Small outward ticks
    axis_ticks=gg.element_line(color="#333333", size=0.3),
    axis_ticks_length=3,
    axis_ticks_direction="out",
    axis_ticks_minor=gg.element_blank(),
    # Text
    axis_text=gg.element_text(size=6, family=FONT_FAMILY, color="#333333"),
    axis_title=gg.element_text(
        size=7, family=FONT_FAMILY, color="#333333", weight="normal"
    ),
    title=gg.element_text(size=7, family=FONT_FAMILY, weight="bold"),
    # Clean background
    panel_background=gg.element_rect(fill="white", color="none"),
    panel_grid_major=gg.element_blank(),
    panel_grid_minor=gg.element_blank(),
    plot_background=gg.element_rect(fill="white", color="none"),
    panel_border=gg.element_blank(),
    # Sizing
    figure_size=(3, 2),
    # Legend
    legend_title=gg.element_text(
        size=6, family=FONT_FAMILY, color="#333333", weight="normal"
    ),
    legend_text=gg.element_text(size=5, family=FONT_FAMILY, color="#333333"),
    legend_key_size=10,
    legend_key=gg.element_rect(fill="white", color="none"),
    legend_background=gg.element_rect(fill="white", color="#CCCCCC", size=0.3),
    legend_margin=0,
    legend_entry_spacing=2,
    # Facet strips
    strip_text=gg.element_text(
        size=6, family=FONT_FAMILY, color="#333333", weight="normal"
    ),
    strip_background=gg.element_rect(fill="white", color="none"),
)
