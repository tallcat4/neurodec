# 新しいコーデックに適用する手順

前提は、エンコーダー CLI（音声 → フレームごとの固定長バイト列）が手に入ることです。デコーダー CLI があれば
評価の基準にします（なくても学習はできます。その場合 `reference` 行が出ないだけです）。実機の無線機なら
「送信音声 → SDR で取り出したビット列」の組を `data/<campaign>/train/*.npz` と同じ形式
（`bits` uint8 [T, S]、`src` int16 クリーン音声、`spans`）で用意すれば、手順 3 以降は同じです。

## 0. 環境

```powershell
.\setup-windows.ps1                 # uv で .venv（Python 3.12、CUDA 版 torch）
.\run-neurodec.ps1 -m pytest -q     # 55 件通過
.\run-neurodec.ps1 -c "import torch; print(torch.cuda.is_available())"
```

コーパス（JVS、JSUT、LibriTTS-R）は `corpora/` に置きます（`scripts/download_corpora.sh`、Windows は
`scripts/unpack_corpora.py`）。ボコーダーを既存のものから流用するなら、新コーデックで必要なのは
キャンペーンに使う音声だけです。

## 1. コーデックを「無線機」に登録する（コードを読まずに）

`neurodec/blackbox/paths.py`: バイナリの場所（ディレクトリ、encoder / decoder の相対パス、環境変数名）です。

```python
"mrl": dict(env="NEURODEC_MRL_DIR", dir="mrl-celp-c", encoder="encoder/wav2mrl", decoder="decoder/mrl2wav"),
```

`neurodec/blackbox/encoder.py` の `FORMATS`: 1 コーデックフレームのバイト数と有効ビット数、20 ms モデル
フレームへの分割数、エンコーダーの引数です。

```python
"mrl80": {"codec": "mrl", "bytes": 20, "bits": 160, "split": 2, "args": []},   # 40 ms = 2 × 20 ms
```

`neurodec/blackbox/radio.py` の `_DECODER_ARGS`: デコーダー CLI の引数（評価専用）です。

これ以外にコーデック固有のコードは要りません。`tests/test_boundary.py` が、学習・解析側のコードが
blackbox を参照していないことを確認します。

## 2. キャンペーンと学習の設定

`configs/<codec>.yaml`（キャンペーン定義）は `configs/celp80.yaml` を手本に、`bits`、`data_dir`、
`blackbox.secret_seed`（置換の seed。学習側は知りません）、`blackbox.block_frames`（バースト長）を決めます。
学習設定は `configs/final/ambe49_decoder.yaml` をコピーし、同じ 4 項目を書き換えます（他は変えません）。
自己完結ファイルなので継承の追跡は不要です。

## 3. ボコーダー（コーデック非依存、通常は流用）

最終版は `runs/mel_istft_excap/last.pt`（推論専用の書き出しは
`checkpoints/final_20261029_clean_speech_vocoder_step30000.pt`）です。新規に学習するなら次のとおりです。

```powershell
.\run-neurodec.ps1 -m scripts.run_stage --job runs/vocoder_job --config configs/final/clean_speech_vocoder.yaml `
    --run runs/vocoder --frontend mel --steps 60000
```

識別器（critics）も本学習の初期値に使うので、`run_stage` が書く `runs/vocoder/last.pt`（D を含む）を
`--vocoder` に渡します。推論専用の書き出しを渡した場合は識別器が新規初期化されます（動きますが、わずかに不利です）。

## 4. 1 コマンドで全段を回す

```powershell
.\run-neurodec.ps1 -m scripts.new_codec_pipeline --config configs/final/<codec>_decoder.yaml `
    --name <codec>_v1 --vocoder runs/mel_istft_excap/last.pt --build --start
```

| 段 | 内容 | 所要（RTX 4070 Ti SUPER、単独） |
|---|---|---|
| build（`--build` 時） | `neurodec.data.build`: 12 h のキャンペーン生成（CPU 16 並列） | 数時間 |
| teachers | `scripts.precompute_f0`: F0 / 有声度 / 帯域非周期性（CPU） | 数分 |
| probe | `scripts.probe_frontend` 20000 step | 約 10 分 |
| joint | `scripts.run_stage` 30000 step ＋ 評価（clean / sim × BER 0, 1, 3 %） | 約 70 分 |
| listen | `scripts.listen_lang` 男女別 6 発話 | 約 2 分 |

進捗は `runs/<name>_pipeline/queue.log` と各 `runs/<name>_*/status.json`（`stage`: training / evaluating /
completed / failed）で分かります。既存の run があれば起動を拒否します（上書き防止）。

手動で段ごとに回すなら、生成された `runs/<name>_pipeline/plan.json` の各 `args` をそのまま
`.\run-neurodec.ps1 -m <module> <args>` に渡せます。`--hours 3` などで録音予算の前方部分だけを使えます。

## 5. 結果を読む

- `results/<name>_joint_job/summary.md`: 条件 × BER × システムの PESQ-NB / STOI / log-mel dB / F0 RMSE / V-UV。
- `results/<name>_joint_job/per_utterance.jsonl` → `.\run-neurodec.ps1 -m scripts.gender_summary --config <config> results/<name>_joint_job`
  で性別・コーパス別に集計できます。
- `results/listen_<name>_{male,female}/`: `1_target`, `2_<codec>-decoder`, `3_proposed_*` の同一発話 WAV。
  試聴セットにはこの 3 種だけを入れます（正解 mel 上限は紛らわしいので `--with-upper-bound` の診断時のみ）。
- 判定の目安: 提案が自身のデコーダーを sim と BER で上回り、clean で 0.2 以内なら AMBE+2 と同水準です。

## 6. 出荷と復号

```powershell
.\run-neurodec.ps1 -m scripts.export_checkpoint --out-json checkpoints/<codec>.json `
    decoder=runs/<name>_joint/step30000.pt:checkpoints/<codec>_decoder.pt
.\run-neurodec.ps1 -m neurodec.infer --ckpt checkpoints/<codec>_decoder.pt --bits capture.npy --out out.wav
```

`capture.npy` は uint8 [T, S] です（無線機から取り出したまま、置換を戻しません）。チェックポイントには
blackbox 節（seed）は入りません。

## 運用上の注意

- GPU の同時学習は 3 本までです。VRAM 14 GB を超えると Windows ドライバがシステムメモリに退避し、全ジョブが止まります。
- 学習を途中で止めるには run ディレクトリに `STOP` ファイルを置きます（次のログ間隔で保存して終了）。再開は
  `STOP` を消して `python -m neurodec.train --config <config> --run runs/<run>` です（`run_stage` は新しい `--job` で）。
- 比較は同じ step 数で行い、速度は単独実行のベンチ（`scripts/benchmark_backends.py`）で測ります。
- 実行していないテスト・学習・評価の結果は書きません。試聴 WAV は必ず同一発話で揃えます。
