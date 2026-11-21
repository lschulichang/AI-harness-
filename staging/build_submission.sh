#!/usr/bin/env bash
# Build, smoke-test, and export the single Linux/amd64 image for the contest.
set -euo pipefail
cd -- "$(dirname -- "$0")"
tag=contest-harness-abc:1.0.2
archive=mirror/4.tar

docker build --platform linux/amd64 --tag "$tag" .
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$tag")" = linux/amd64
docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp "$tag" describe |
    python3 -c '
import json, sys
description = json.load(sys.stdin)
assert description["protocol"] == "contest.harness.workspace.v2"
assert description["model_access"] == "host_gateway_rpc"
assert description["sdk_version"] == "1.0.2"
assert set(description["tasks"]) == {"classification", "regression"}
print(json.dumps(description))
'
mkdir -p -- "$(dirname -- "$archive")"
pending=$(mktemp "./${archive}.XXXXXXXX")
trap 'rm -f -- "$pending"' EXIT
docker image save -o "$pending" "$tag"
python3 - "$pending" <<'PY'
import json
from pathlib import Path
import sys
import tarfile

path = Path(sys.argv[1])
if path.stat().st_size > 2_000_000_000:
    raise SystemExit("比赛镜像归档超过 2 GB")
with tarfile.open(path, "r:") as archive:
    entries = json.load(archive.extractfile("manifest.json"))
    if len(entries) != 1 or entries[0]["RepoTags"] != ["contest-harness-abc:1.0.2"]:
        raise SystemExit("镜像归档须包含单个比赛镜像")
    config = json.load(archive.extractfile(entries[0]["Config"]))
    if (config["os"], config["architecture"]) != ("linux", "amd64"):
        raise SystemExit("镜像平台应为 Linux/amd64")
    if config["config"]["Entrypoint"] != ["python3", "/app/harness.py"]:
        raise SystemExit("镜像入口与比赛 SDK 不一致")
PY
chmod 0644 "$pending"
mv -f -- "$pending" "$archive"
sha256sum "$archive"
