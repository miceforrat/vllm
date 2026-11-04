#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point: the implementation remains in the original A/B
# script while the experiment has expanded to a three-mode matrix.
exec bash "$(dirname "$0")/run_reasoner_breakable_ab.sh" "$@"
