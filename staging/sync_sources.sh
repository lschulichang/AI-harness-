#!/usr/bin/env bash
# Refresh the allow-listed runtime files from the sibling development project.
set -euo pipefail
target=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$target/../AI+harness" && pwd)

copy() {
    install -D -m 0644 "$project/$1" "$target/${2:-$1}"
}

for module in config credentials runner llm_client tracking storage portability resource_monitor \
              gas_runner gas_data gas_drift gas_metrics gas_policy \
              gas_training gas_workflow feature_drift \
              rul_policy rul_workflow __init__; do
    copy "harness_core/$module.py"
done
for module in __init__ gas_dataset bearing_baseline bearing_pairs cmapss_baseline rul_baseline; do
    copy "wrappers/$module.py"
done
for bundle in gas cpu-b-rul-20261110-v1 cpu-c-guide-20261110-v1; do
    for module in __init__ model data train predict; do
        copy "baseline/$bundle/baselines/$module.py"
    done
done
copy "baseline/cpu-b-rul-20261110-v1/baselines/configs/B.json"
copy "baseline/cpu-c-guide-20261110-v1/baselines/configs/C.json"

sdk=contest-harness-sdk-1.0.2
copy "$sdk/template/harness.py" "harness.py"
copy "$sdk/pyproject.toml" "sdk/pyproject.toml"
copy "$sdk/README.md" "sdk/README.md"
for src in "$project/$sdk"/contest_harness/*.py; do
    copy "$sdk/contest_harness/${src##*/}" "sdk/contest_harness/${src##*/}"
done
