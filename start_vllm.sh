#!/bin/bash
# Start vLLM with monkey-patch for prometheus_fastapi_instrumentator bug

python3 -c "
import sys, prometheus_fastapi_instrumentator.routing as r
orig = r._get_route_name
def _patched(scope, routes):
    try:
        return orig(scope, routes)
    except AttributeError:
        return 'unknown'
r._get_route_name = _patched
from vllm.entrypoints.cli.main import main
sys.argv = ['vllm', 'serve'] + sys.argv[1:]
main()
" "$@"
