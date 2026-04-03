#!/usr/bin/env bash
set -euo pipefail
cd /home/zhahl/LDMVFI
mkdir -p dist
LOG_FILE="dist/pack_ldmvfi_$(date +%Y%m%d_%H%M%S).log"
nohup /home/zhahl/miniconda3/envs/ldmvfi/bin/python -u - <<'PY' > "$LOG_FILE" 2>&1 &
import conda_pack
out='/home/zhahl/LDMVFI/dist/ldmvfi_env_api.tar.gz'
print('packing to', out, flush=True)
res = conda_pack.pack(prefix='/home/zhahl/miniconda3/envs/ldmvfi', output=out, format='tar.gz', force=True, compress_level=4, n_threads=1, ignore_missing_files=True, ignore_editable_packages=True, verbose=True)
print('result', res, flush=True)
PY
echo "$! $LOG_FILE"
