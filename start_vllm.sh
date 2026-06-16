#!/bin/bash
# Start vLLM with monkey-patch for prometheus_fastapi_instrumentator bug
# The library assumes all routes have .path, but FastAPI _IncludedRouter doesn't.

python3 -c "
import prometheus_fastapi_instrumentator.routing as r
orig = r._get_route_name
def _patched(scope, routes):
    try:
        return orig(scope, routes)
    except AttributeError as e:
        # '_IncludedRouter' object has no attribute 'path'
        return 'unknown'
r._get_route_name = _patched
" && exec vllm serve "$@"
