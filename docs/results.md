# 評価結果

評価プロトコル: 8 kHz、PESQ-NB / STOI / log-mel 誤差 dB / F0 RMSE cent / V-UV 誤り率です。乱数 seed は送信とモデルごとに
固定し、同一評価内で比較します。bits49 テストセットは JVS・LibriTTS-R の未知話者 20 名、364 発話です。
比較対象はコーデック自身のデコーダー（ambe2-c の `ambe2wav`、mrl-celp-c の `mrl2wav`）です。

## 1. AMBE+2 half-rate（49 bit / 20 ms）

設定 `configs/final/ambe49_decoder.yaml`、チェックポイント `checkpoints/final_20261029_ambe49_decoder_step30000.pt`、
評価出力 `results/joint_b49_wide_bridge_f0ap_job/summary.md`。

| 条件 | BER | neurodec PESQ / STOI | ambe2-c PESQ / STOI |
|---|---|---|---|
| clean | 0 | 2.587 / 0.875 | 2.778 / 0.902 |
| clean | 1 % | 2.468 / 0.865 | 2.152 / 0.843 |
| clean | 3 % | 2.225 / 0.843 | 1.651 / 0.752 |
| 音響シミュレーション | 0 | 2.398 / 0.855 | 2.152 / 0.807 |
| 音響シミュレーション | 1 % | 2.300 / 0.846 | 1.811 / 0.754 |
| 音響シミュレーション | 3 % | 2.101 / 0.825 | 1.522 / 0.674 |

性別・コーパス別（clean、BER 0、`scripts/gender_summary.py`）:

| 話者群 | n | neurodec | ambe2-c |
|---|---|---|---|
| JVS 女性 | 119 | 2.611 | 2.585 |
| JVS 男性 | 50 | 2.561 | 2.856 |
| LibriTTS-R 女性 | 61 | 2.646 | 2.827 |
| LibriTTS-R 男性 | 134 | 2.548 | 2.897 |

主観評価（`results/listen_ja_male_f0ap`, `listen_ja_female_f0ap`）: 男女ともに実用的な品質です。女性話者は
ambe2-c を上回ります。男性話者では ambe2-c の方が自然に聞こえる場面があります。

## 2. CELP（mrl-celp-c、160 bit / 40 ms → 80 bit × 2 フレーム）

設定 `configs/final/celp80_decoder.yaml`（周期性ヘッドなし）、チェックポイント
`checkpoints/final_20261029_celp80_decoder_step30000.pt`、評価出力 `results/joint_celp80_wide_bridge_job`。
AMBE+2 からの変更は format 1 行とデコーダーのコマンド行 1 行だけです。

| 条件 | BER | neurodec | mrl-celp-c |
|---|---|---|---|
| clean | 0 | 2.181 / 0.824 | 2.734 / 0.886 |
| clean | 3 % | 1.900 / 0.788 | 1.353 / 0.585 |
| 音響シミュレーション | 0 | 2.118 / 0.815 | 2.309 / 0.805 |

周期性ヘッド付きの設定は CELP ではまだ評価していません。

## 3. 先行手法との比較（同じデータ・同じ step）

- フロントエンド単体（ビット → log-mel 回帰 probe）: BitToken は先行手法の dequantizer より 1.2 dB 良いです。
- 学習 20000 step、2 バックエンド × 2 フロントエンド（`results/joint_controls_20k`）: BitToken が両バックエンドで PESQ +0.25 です。
- 先行手法の構成（1 時間データ、PWG 46k step ＋ MPD/MRD 継続）は clean 1.98 / 0.80、本構成は 2.59 / 0.875 です。

## 4. 上限と律速

- 正解 mel からの再合成（クリーン音声ボコーダー、`results/mel_istft_excap_job`）: 3.68 / 0.96 / 2.1 dB。
  受信ビット条件（2.59）との差はフロントエンドの mel 誤差（5.3 dB、ambe2-c は 4.4 dB）です。
- 学習データ量（小型構成）: 1 / 3 / 6 / 12 h で 2.12 / 2.26 / 2.37 / 2.33。3〜6 時間で飽和します。

## 5. 速度（単独ベンチマーク、batch 12 × 1 秒、G+D 全更新、`results/frontend_benchmark.json`）

| 構成 | 更新 ms | 推論（12 秒分）ms | peak GiB |
|---|---|---|---|
| 先行手法（dequantizer ＋ PWG ＋ MPD/MRD） | 120 | 46.5 | 2.66 |
| BitToken（MLP、幅 256）＋ iSTFT | 69 | 2.6 | 1.41 |

幅 384 / 8 ブロックのフロントエンドは単独ベンチマーク未計測です（学習 30000 step は単独で約 55 分）。

## 6. 男性話者の差について

| 仮説 | 検証 | 結果 |
|---|---|---|
| ピッチ追跡の誤り | RPA/RCA（`scripts/diagnostics/pitch_accuracy.py`） | 倍半誤りなし。neurodec の RPA は ambe2-c より高い |
| 励振の与え方 | 正弦波和・cyclic noise・帯域別・Wavehax 型を追加 | 男性 +0.01〜0.05。感度試験でボコーダーが励振を無視していた |
| ボコーダー上限 | 正解 mel 入力の男女別 PESQ | 男性 3.57 / 女性 3.85。iSTFT 系、PWG とも同じ差 |
| 事前学習データの男女比 | LibriTTS-R 追加＋等確率サンプリング | 変化なし |
| 時間分解能 | hop 5 ms | 変化なし |
| 周期性を構成で強制 | ソースフィルタ合成（硬・柔） | 男女差は 0.1 に縮むが上限 3.26 に低下、学習後 2.44 |

差はピッチ追跡ではなくボコーダーの微細構造に由来します。本構成では未解決です。
