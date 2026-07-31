"""Control plane: places calls and serves the webhooks that steer them.

Everything here STEERS calls. Nothing here produces a reported number -- those
come exclusively from `analyzer/` reading saved artifacts. If you find
yourself computing a latency in this package, stop.
"""
