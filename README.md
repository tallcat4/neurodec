# neurodec — ブラックボックス音声コーデックのニューラルデコーダー

未知の低レート音声コーデックのデコーダーを、**「クリーン音声を送信する → 受信したビット列を取り出す」組だけ**から
学習します。コーデック内部（ビット割当、コードブック、インターリーブ）は使いません。
AMBE+2 half-rate（49 bit / 20 ms）と CELP（160 bit / 40 ms）で同じ手順を検証しました。

## 結果（未知話者 20 名、364 発話、PESQ-NB / STOI）

| 条件 | 提案 | コーデック自身のデコーダー |
|---|---|---|
| AMBE+2, clean | 2.59 / 0.88 | 2.78 / 0.90 |
| AMBE+2, ビット誤り 3 % | **2.23 / 0.84** | 1.65 / 0.75 |
| AMBE+2, 音響シミュ（部屋・AGC・雑音） | **2.40 / 0.86** | 2.15 / 0.81 |
| CELP, clean | 2.18 / 0.82 | 2.73 / 0.89 |
| CELP, ビット誤り 3 % | **1.90 / 0.79** | 1.35 / 0.59 |

試聴では男女ともに実用的と判断しています（2026-10-29）。clean 条件の PESQ は自身のデコーダーに 0.2 届かず、
男性話者で差が大きいです。出発点の論文手法（Kim et al., IEEE SPL 2022）に対しては、同条件で PESQ +0.25、推論は 18 倍速です。

## 仕組み

```
受信ビット列 [T, S] → BitToken フロントエンド → 予測 log-mel（＋F0 / 有声度 / 帯域非周期性）→ 凍結 iSTFT ボコーダー → 音声
```

- **BitToken フロントエンド**: ビット位置ごとの埋め込み → フレーム内 MLP → 時間方向 ConvNeXt。ビットの順序や隣接性を仮定しません。
- **ボコーダーはクリーン音声だけで事前学習**して凍結します。コーデックごとに学習するのはフロントエンドだけです。
- **補助ヘッド**（log-F0、有声度、5 帯域の非周期性）と **ビット誤り拡張**（0〜2 %）が品質と頑健性の源です。

## 使い方

```powershell
.\setup-windows.ps1                                                     # uv で .venv（CUDA 版 torch）
.\run-neurodec.ps1 -m pytest -q                                          # 55 件
.\run-neurodec.ps1 -m scripts.new_codec_pipeline --config configs/final/<codec>_decoder.yaml `
    --name <codec>_v1 --vocoder runs/mel_istft_excap/last.pt --build --start   # 生成→教師→probe→本学習→評価→試聴
.\run-neurodec.ps1 -m neurodec.infer --ckpt checkpoints/final_20261029_ambe49_decoder_step30000.pt `
    --bits capture.npy --out out.wav                                     # 受信ビット列 uint8 [T, S] を復号
```

新しいコーデックの登録は 3 行（エンコーダー CLI の形式、デコーダー CLI の引数、バイナリの場所）と設定ファイル 1 つです。
手順は [docs/new_codec.md](docs/new_codec.md) にあります。所要時間（RTX 4070 Ti SUPER）は probe 約 10 分、
本学習 30000 step 約 60 分、評価 約 10 分、ボコーダーの新規学習 60k step 約 1.5 時間です。

## 文書

| 読みたいこと | 場所 |
|---|---|
| 手法と設計判断 | [docs/method.md](docs/method.md) |
| 新しいコーデックへの適用手順 | [docs/new_codec.md](docs/new_codec.md) |
| 全数値・性別別・試聴の判定 | [docs/results.md](docs/results.md) |
| 効かなかったこと | [docs/experiments.md](docs/experiments.md) |
| 参考文献 | [docs/references.md](docs/references.md) |

## 成果物と配置

| もの | 場所 |
|---|---|
| 最終デコーダー（AMBE+2 49 bit、推論専用） | `checkpoints/final_20261029_ambe49_decoder_step30000.pt` |
| クリーン音声ボコーダー（コーデック非依存） | `checkpoints/final_20261029_clean_speech_vocoder_step30000.pt` |
| CELP デコーダー | `checkpoints/final_20261029_celp80_decoder_step30000.pt` |
| SHA256 と出自 | `checkpoints/final_20261029.json` |
| 最終設定（自己完結） | `configs/final/ambe49_decoder.yaml`, `clean_speech_vocoder.yaml`, `celp80_decoder.yaml` |

```
neurodec/              ライブラリ
  blackbox/            「無線機」: エンコーダー CLI ＋ 秘密インターリーバー ＋ 評価用デコーダー。学習側はここを見ません
  data/                キャンペーン生成（音響シミュ、一定遅延の推定）、データセット、F0 / 非周期性の教師
  models/              frontends.py (BitToken), istft.py (ボコーダー), excitation.py, discriminators.py,
                       dequant.py / pwg.py (論文手法の再現)
  eval/                PESQ-NB / STOI / log-mel / F0 / V-UV、同一発話 WAV
  train.py, infer.py, losses.py, config.py
configs/               キャンペーン定義（base, bits49, bits72, celp80, smoke）、final/（最終レシピ）、experiments/（検討過程）
scripts/               new_codec_pipeline.py と各段（precompute_f0, probe_frontend, run_stage, overnight_queue, listen_lang）、
                       gender_summary, export_checkpoint, assemble_decoder, benchmark_backends、diagnostics/（原因分析）
tests/                 55 件（ブラックボックス境界テストを含む）
checkpoints/           出荷用チェックポイント
docs/                  method, new_codec, results, experiments, references
corpora/, data/, runs/, results/   ローカル専用（Git 管理外）。別マシンで推論するには checkpoints/ と configs/final/ で足ります
```

## 制約

- コーデックはブラックボックスです。エンコーダー CLI で学習入力を作り、デコーダー CLI は評価基準にだけ使います。ソースは読みません。
- インターリーバーの置換と seed は `neurodec/blackbox/` の外に出しません（`tests/test_boundary.py`）。
- 録音予算は 12 時間以内、無線遅延は一定として 1 回だけ推定します。
- 既存の run・結果・チェックポイントは上書きしません。GPU の同時学習は 3 本までです（VRAM 14 GB 超で全ジョブが止まります）。
- 学習を途中で止めるには run ディレクトリに `STOP` ファイルを置きます。

## ライセンス

- このリポジトリのコードと文書は **GNU General Public License v3.0 or later**（[LICENSE](LICENSE)）です。
  コードはすべてこのプロジェクトで書いたもので、第三者のコードは含みません。依存ライブラリ（torch, numpy, scipy,
  soundfile, PyYAML, tensorboard, pesq, pystoi, pyworld）はいずれも GPL と両立する許諾（BSD / MIT / Apache-2.0）です。
- コーデック（`ambe2-c`, `mrl-celp-c`）は含みません。別プロセスの CLI として呼び出すだけで、入手と利用条件は各自で確認してください。
- 参照した論文の PDF は含みません（出典は `docs/references.md`）。
- `checkpoints/` の重みは JVS・JSUT（学術・非商用の研究利用、再配布不可）と LibriTTS-R（CC BY 4.0）の音声で学習しました。
  重みの利用はこれらコーパスの条件に従ってください。商用利用には JVS / JSUT の権利者の許諾が必要です。
