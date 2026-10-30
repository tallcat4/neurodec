#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Downloads the corpora into $CORPORA (default: corpora).
#   LibriTTS-R train-clean-100 / test-clean (CC BY 4.0, openslr 141)
#   JSUT ver1.1 (single female speaker; research / commercial-with-terms, see readme)
#   JVS ver1 (100 speakers; Google Drive, needs gdown; terms in the archive)
set -euo pipefail
CORPORA=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
mkdir -p "$CORPORA" && cd "$CORPORA"

fetch() {  # url file
  [ -f "$2.done" ] && return
  curl -L --fail --retry 5 -C - -o "$2" "$1"
  touch "$2.done"
}

for sub in train_clean_100 test_clean; do
  fetch "https://www.openslr.org/resources/141/${sub}.tar.gz" "libritts_r_${sub}.tar.gz"
  [ -d LibriTTS_R/${sub//_/-} ] || tar xzf "libritts_r_${sub}.tar.gz"
done

fetch "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip" jsut_ver1.1.zip
[ -d jsut_ver1.1 ] || unzip -q jsut_ver1.1.zip

if [ ! -d jvs_ver1 ]; then
  if [ ! -f jvs_ver1.zip ]; then
    uvx gdown "https://drive.google.com/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt" -O jvs_ver1.zip \
      || echo "JVS: automatic download failed. Get jvs_ver1.zip from https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus and put it in $CORPORA"
  fi
  [ -f jvs_ver1.zip ] && unzip -q jvs_ver1.zip
fi
du -sh "$CORPORA"/*/ 2>/dev/null
