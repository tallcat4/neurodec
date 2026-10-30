# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest
import soundfile as sf

from neurodec.data.corpus import Utterance
from neurodec.data.dataset import CleanCropDataset
from neurodec.train import gender_balance_weights


def _utts(tmp_path):
    out = []
    for i, (spk, amp) in enumerate([('jvs/jvs001', 0.5), ('jvs/jvs002', 0.0)]):
        f = tmp_path / f'{i}.wav'
        sf.write(f, np.full(8000, amp, np.float32) if amp else np.zeros(8000, np.float32), 8000)
        out.append(Utterance(str(f), spk, 'ja', 'jvs', 1.0))
    return out


def test_clean_crops_follow_sampling_weights(tmp_path):
    utts = _utts(tmp_path)
    ds = CleanCropDataset(utts, 1600, items_per_epoch=20, seed=0, weights=[1.0, 0.0])
    assert all(ds[k][1].abs().sum() > 0 for k in range(20))  # never the silent second utterance
    with pytest.raises(ValueError):
        CleanCropDataset(utts, 1600, weights=[1.0])
    with pytest.raises(ValueError):
        CleanCropDataset(utts, 1600, weights=[0.0, 0.0])


def test_gender_balance_weights(tmp_path, monkeypatch):
    utts = _utts(tmp_path) + [Utterance('x', 'jvs/jvs003', 'ja', 'jvs', 2.0)]
    monkeypatch.setattr('neurodec.data.corpus.speaker_genders',
                        lambda entries: {'jvs/jvs001': 'M', 'jvs/jvs002': 'F', 'jvs/jvs003': 'F'})
    cfg = {'corpora': [], 'mel_pretrain': {'balance_gender': True}}
    w = gender_balance_weights(cfg, utts)
    # M: 1 s, F: 3 s -> male utterances weighted 3x so both genders get equal total probability
    assert np.isclose(w[0] * 1.0, w[1] * 1.0 + w[2] * 2.0)
    assert gender_balance_weights({'corpora': []}, utts) is None
