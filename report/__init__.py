"""Report rendering. Pure presentation -- reads saved artifacts, computes nothing."""

from .html_report import render_html, write_html

__all__ = ["render_html", "write_html"]
