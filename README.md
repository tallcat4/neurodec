# neurodec — ブラックボックス音声コーデックのニューラルデコーダー

未知の低レート音声コーデックのデコーダーを、**送信したクリーン音声と受信したビット列の組だけ**から学習します。
コーデック内部の情報（ビット割当、コードブック、インターリーブ）は使いません。
AMBE+2 half-rate（49 bit / 20 ms）と CELP（160 bit / 40 ms）の 2 系統で同じ手順を検証しています。

## 結果

未知話者 20 名・364 発話、8 kHz、PESQ-NB / STOI。比較対象はコーデック自身のデコーダーです。

| 条件 | neurodec | コーデック自身のデコーダー |
|---|---|---|
| AMBE+2, clean | 2.59 / 0.88 | 2.78 / 0.90 |
| AMBE+2, ビット誤り 3 % | **2.23 / 0.84** | 1.65 / 0.75 |
| AMBE+2, 音響シミュレーション（部屋・AGC・雑音） | **2.40 / 0.86** | 2.15 / 0.81 |
| CELP, clean | 2.18 / 0.82 | 2.73 / 0.89 |
| CELP, ビット誤り 3 % | **1.90 / 0.79** | 1.35 / 0.59 |

ビット誤りと音響劣化のある条件ではコーデック自身のデコーダーを上回り、clean 条件では PESQ で 0.2 下回ります。
差は男性話者で大きいです。先行手法（Kim et al., IEEE SPL 2022）と同じデータ・同じ学習 step で比べると PESQ +0.25、
推論は約 18 倍速です。詳細は [docs/results.md](docs/results.md) にあります。

## 方式

```
受信ビット列 [T, S] → BitToken フロントエンド → log-mel（＋F0 / 有声度 / 帯域別非周期性）→ iSTFT ボコーダー（凍結）→ 音声
```

- **BitToken フロントエンド**: ビット位置ごとの埋め込み → フレーム内 MLP → 時間方向 ConvNeXt。ビットの順序や隣接性を仮定しません。
- **ボコーダー**はクリーン音声だけで事前学習し、凍結します。コーデックごとに学習するのはフロントエンドだけです。
- **補助ヘッド**（log-F0、有声度、5 帯域の非周期性）と学習時の**ビット誤り付加**（0〜2 %）で品質と頑健性を得ています。
- **因果モード**（`model.causal`、任意で 1 フレームの先読み）で、受信しながら逐次に復号できる低遅延のデコーダーも学習できます。
  TOKUDER の実機データでは、経験的に、先読み 1 フレーム（20 ms）で非因果と同等以上の品質（PESQ 2.53 対 2.46）を、
  半分以下のパラメータで得ています（[docs/experiments.md](docs/experiments.md#因果モードtokuder-での経験)）。

設計の根拠は [docs/method.md](docs/method.md)、比較した代替案は [docs/experiments.md](docs/experiments.md) にあります。

## 使い方

```powershell
.\setup-windows.ps1                                                     # uv で .venv を作成（CUDA 版 torch）
.\run-neurodec.ps1 -m pytest -q
.\run-neurodec.ps1 -m scripts.new_codec_pipeline --config configs/final/<codec>_decoder.yaml `
    --name <codec>_v1 --vocoder runs/mel_istft_excap/last.pt --build --start   # データ生成 → 教師 → probe → 学習 → 評価 → 試聴セット
.\run-neurodec.ps1 -m neurodec.infer --ckpt checkpoints/final_20261029_ambe49_decoder_step30000.pt `
    --bits capture.npy --out out.wav                                     # 受信ビット列 uint8 [T, S] を復号
```

新しいコーデックの追加は、エンコーダー CLI の形式、デコーダー CLI の引数、バイナリの場所の 3 行と、設定ファイル 1 つです。
手順は [docs/new_codec.md](docs/new_codec.md) にあります。所要時間の目安（RTX 4070 Ti SUPER）は、probe 約 10 分、
学習 30000 step 約 60 分、評価 約 10 分、ボコーダーの新規学習 60k step 約 1.5 時間です。

## 文書

| 内容 | 場所 |
|---|---|
| 方式と設計判断 | [docs/method.md](docs/method.md) |
| 新しいコーデックへの適用手順 | [docs/new_codec.md](docs/new_codec.md) |
| 評価結果（条件別・性別別・主観評価） | [docs/results.md](docs/results.md) |
| 比較した代替案 | [docs/experiments.md](docs/experiments.md) |
| 参考文献 | [docs/references.md](docs/references.md) |

## 配布物と構成

| 内容 | 場所 |
|---|---|
| AMBE+2（49 bit）デコーダー、推論用 | `checkpoints/final_20261029_ambe49_decoder_step30000.pt` |
| クリーン音声ボコーダー（コーデック非依存） | `checkpoints/final_20261029_clean_speech_vocoder_step30000.pt` |
| CELP（160 bit）デコーダー | `checkpoints/final_20261029_celp80_decoder_step30000.pt` |
| SHA256 | `checkpoints/final_20261029.json` |
| 学習設定（自己完結） | `configs/final/ambe49_decoder.yaml`, `clean_speech_vocoder.yaml`, `celp80_decoder.yaml` |

```
neurodec/              ライブラリ
  blackbox/            コーデックのエンコーダー CLI 呼び出し、秘密インターリーバー、評価用デコーダー。学習側からは参照しません
  data/                データ生成（音響シミュレーション、一定遅延の推定）、データセット、F0 / 非周期性の教師
  models/              frontends.py (BitToken), istft.py (ボコーダー), excitation.py, discriminators.py,
                       dequant.py / pwg.py (先行手法)
  eval/                PESQ-NB / STOI / log-mel / F0 / V-UV、同一発話の WAV 出力
  train.py, infer.py, losses.py, config.py
configs/               データ生成の定義（base, bits49, bits72, celp80, smoke）、final/（学習設定）、experiments/（比較に使った設定）
scripts/               new_codec_pipeline.py と各段（precompute_f0, probe_frontend, run_stage, overnight_queue, listen_lang）、
                       gender_summary, export_checkpoint, assemble_decoder, benchmark_backends、diagnostics/（分析ツール）
tests/                 62 件（ブラックボックス境界、因果モードの未来非依存のテストを含む）
checkpoints/           推論用チェックポイント
docs/                  method, new_codec, results, experiments, references
corpora/, data/, runs/, results/   ローカル専用（Git 管理外）。推論だけなら checkpoints/ と configs/final/ で足ります
```

## 前提と制約

- コーデックはブラックボックスとして扱います。エンコーダー CLI で学習入力を作り、デコーダー CLI は評価の基準にだけ使います。
- インターリーバーの置換と seed は `neurodec/blackbox/` の外に出ません（`tests/test_boundary.py`）。
- 送受信ペアの録音は 12 時間以内、無線遅延は一定として 1 回だけ推定します。
- GPU の同時学習は 3 本までを目安にしてください（VRAM 14 GB 超で全ジョブが大幅に遅くなります）。
- 学習を途中で止めるには run ディレクトリに `STOP` ファイルを置きます。

## ライセンス

- コードと文書は **GNU General Public License v3.0 or later**（[LICENSE](LICENSE)）です。
  コードはすべて本プロジェクトで書いたもので、第三者のコードは含みません。依存ライブラリ（torch, numpy, scipy,
  soundfile, PyYAML, tensorboard, pesq, pystoi, pyworld）は GPL と両立する許諾（BSD / MIT / Apache-2.0）です。
- コーデック（`ambe2-c`, `mrl-celp-c`）は含みません。別プロセスの CLI として呼び出すだけです。入手と利用条件は各自で確認してください。
- 参照した論文の PDF は含みません（出典は `docs/references.md`）。
- `checkpoints/` の重みは JVS・JSUT（学術・非商用の研究利用、再配布不可）と LibriTTS-R（CC BY 4.0）の音声で学習しています。
  重みの利用はこれらコーパスの条件に従ってください。商用利用には JVS / JSUT の権利者の許諾が必要です。
