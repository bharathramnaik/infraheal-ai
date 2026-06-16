#!/bin/bash
# Start vLLM with monkey-patch for prometheus_fastapi_instrumentator bug
# The library assumes all routes have .path, but FastAPI _IncludedRouter doesn't.

python3 -c "
import sys, prometheus_fastapi_instrumentator.routing as r
orig = r._get_route_name
def _patched(scope, routes):
    try:
        return orig(scope, routes)
    except AttributeError:
        return 'unknown'
r._get_route_name = _patched
# Now import and run vLLM CLI main in the same process
from vllm.entrypoints.cli.main import main
sys.argv = ['vllm'] + sys.argv[1:]
main()
" "$@"
