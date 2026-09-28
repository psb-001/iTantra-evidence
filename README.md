# iTantra — evidence for SIH26173

Every number claimed in our SIH26173 submission, and the file in this repository that
produces it. Nothing here is a summary written by hand: each figure is read from the file
beside it, and each file is raw output from a script in the main repository.

Problem statement: **SIH26173** — Indian Multilingual TTS & STT Aided Neural Transceiver
Radio Access for low-bitrate links.

---

## Accuracy — 10 of 10 required languages

**Source: `eval/SUMMARY_full_v3pruned.json`**

| Language | Word error | Letter error |
|---|---|---|
| Kannada | 0.307 | 0.053 |
| Malayalam | 0.361 | 0.059 |
| English | 0.181 | 0.067 |
| Gujarati | 0.300 | 0.071 |
| **Hindi** | **0.269** | **0.096** |
| Bengali | 0.359 | 0.100 |
| Telugu | 0.431 | 0.100 |
| Marathi | 0.416 | 0.108 |
| Tamil | 0.501 | 0.129 |
| Odia | 0.509 | 0.134 |

- **2 000 sentences, 200 per language, 6.72 hours** of real recorded human speech
- Mean letter error **0.0918** — about 9 wrong letters in every 100, or 1 in 11
- Every language under 0.14

Both "letter error" and "word error" are reported because word error alone flatters
languages with long written words. Normalisation strips punctuation and case.

**Benchmark:** [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0),
used for evaluation only. No audio from FLEURS is redistributed here.

---

## The vocabulary fix — our main technical contribution

**Source: `eval/SUMMARY_full_v2base.json`** (before) vs **`eval/SUMMARY_full_v3pruned.json`**
(after) · script: **`eval/prune_vocab.py`**

The recogniser is character-by-character and takes no language argument, so it substituted
phonetically identical characters from other scripts — "hello" came back as `هello` in Arabic
letters. Our first filter deleted the odd character and transmitted the remainder, so a phone
displayed **"ello"** as if it were correct.

Rather than tune the model, we deleted the **9 286 output characters** the ten languages never
need: the vocabulary went from **10 288 to 1 002**, so the mistake became impossible.

| | Before | After |
|---|---|---|
| Hindi letter error | 0.113 | **0.096** |
| Hindi in the right script, 15 dB background noise | **40%** | **78%** |
| Wrong-script output | 2 in 19 test phrases | **0** |
| The other nine languages | — | **bit-identical** |

The fix cost nothing: nine of ten languages produced byte-for-byte identical output.

---

## Background noise

**Source: `eval/noise_robustness_v3pruned.json`**

Hindi, 50 sentences per condition, white and pink noise at 30/20/15/10/5/0 dB.

| Condition | Letter error | Right script |
|---|---|---|
| Clean | 0.0953 | 100% |
| Pink 15 dB | 0.3935 | 78% |
| Pink 10 dB | 0.4242 | 74% |
| White 0 dB | 0.4877 | 84% |

**This is a limitation, stated plainly:** the noise is synthetic — added to clean recordings
afterwards. It measures sensitivity to noise, not accuracy in a real disaster. We have never
recorded anyone speaking in a real disaster.

---

## Speech output — 7 of 10 languages, 3 text-only

**Source: `eval/tts_intelligibility_pruned.json`**

| Voice | Round-trip letter error |
|---|---|
| English | 0.043 |
| Malayalam | 0.048 |
| Hindi | 0.102 |
| Bengali | 0.106 |
| Telugu | 0.132 |
| Marathi | 0.139 |

**The caveat matters more than the numbers.** This is not a human listening panel. It is the
recognition model reading audio back in, which measures whether the words survive synthesis. A
machine reading its own audio back is not the same as a frightened person understanding it. We
have not run a human panel and we do not claim one.

Gujarati, Kannada and Odia have **no speech output at all**. We could not find a free,
good-quality, offline voice for them that we were permitted to ship. Those three display the
corrected text and stay silent, and the app says so rather than pretending.

---

## On-device measurements

**Source: `app/evidence/pipeline_pruned_model_2026-09-27.txt`**

Real packets captured from the running app:

```
seg0 6.94s  packet=211B
seg1 1.50s  packet=139B
seg2 0.83s  packet=129B
seg0 7.67s  packet=278B   Hindi
```

**Payload:** 6.94 s of speech → **211 bytes**. That is **~66× smaller** than the same sentence
at 16 kbps through a speech codec, and ~1 050× smaller than the raw audio a microphone records.

**On why 66× and not 1 000×:** the 1 000× figure is only true against uncompressed audio, and
only when labelled as such. Against a speech codec — what a damaged radio would actually carry —
the honest figure is ~66×. Reaching 480× would require assuming 117 kbps audio, which is not a
low-bitrate link.

| Measurement | Value | Source |
|---|---|---|
| On-device recognition, real-time factor | 0.147 | `pipeline_pruned_model_2026-09-27.txt` |
| 6.94 s of speech, end to end | 1 020 ms | same |
| Speech synthesis per sentence | 49–380 ms, median 193 | `benchmark_onddevice.txt`, `loopback_test_verified.txt` |
| Memory with the recogniser loaded | 428 MB PSS / 535 MB RSS | `benchmark_onddevice.txt` |
| Idle footprint | 629 KB (voice detector only) | `benchmark_onddevice.txt` |
| 16 KB page-size compatible | yes | see note below |

### A correction worth stating

An earlier version of our evidence claimed this was verified with `zipalign -c -P 16`. Neither
build-tools version we have supports the `-P` flag, so that command could not have produced the
result attributed to it. It is verified instead by inspecting the archive: with
`useLegacyPackaging = true` both native libraries are stored compressed, so no uncompressed
library needs to begin on a 16 KB boundary.

```python
import zipfile
z = zipfile.ZipFile('iTantra-release.apk')
print([(i.filename, i.compress_type) for i in z.infolist() if i.filename.endswith('.so')])
# compress_type 0 = STORED (the failing case), 8 = DEFLATE (the fix)
```

We would rather publish a corrected claim than a convenient one.

---

## What is NOT in this repository, and why

- **The app.** 804 MB, arm64. Distributed by direct link, not through a repository.
- **Model weights.** The recogniser is 340 MB and the voices 448 MB.
- **FLEURS audio.** Evaluation only; the dataset is CC-BY-4.0 and is not redistributed.
- **A human listening panel for the voices.** Not run. Not claimed.

---

## Licences

| Component | Licence | Note |
|---|---|---|
| Meta Omnilingual ASR model | **CC-BY-NC-4.0** | Free to use and modify, **not for commercial use** |
| Hindi Piper voice (pratham) | **CC-BY-NC-SA-4.0** | **Not for commercial use**, share-alike |
| Other Piper voices | MIT wrapper, per-voice terms | Each voice carries its own data licence |
| Silero VAD | MIT | |
| sherpa-onnx, ONNX Runtime | Apache-2.0 / MIT | |
| espeak-ng | GPL-3.0 | Phoneme dictionaries only; not used for synthesis |
| Material Design 3 | Apache-2.0 | |
| FLEURS (evaluation) | CC-BY-4.0 | Not shipped |

**Non-commercial components do ship in our app.** CC-BY-NC-4.0 and CC-BY-NC-SA-4.0 permit use
and modification for non-commercial purposes. We have complied with the attribution and
share-alike conditions, and the app states them in plain language under About → Open-source
credits rather than in a footnote.

---

## Reproducing these numbers

Every file here is raw script output. The scripts that produced them are in the team's private
repository; on request we will share them, and we are happy to run any of them live for an
evaluator.

| File | Produced by |
|---|---|
| `eval/SUMMARY_full_v3pruned.json` | the pruned-vocabulary accuracy run |
| `eval/SUMMARY_full_v2base.json` | the same run before pruning, for comparison |
| `eval/noise_robustness_v3pruned.json` | the noise sweep |
| `eval/tts_intelligibility_pruned.json` | the synthesis round-trip |
| `eval/prune_vocab.py` | the 10 288 → 1 002 vocabulary prune |
| `app/evidence/*.txt` | `adb logcat` from the running app |

## Two-phone result

On 27 September 2026 the team ran the release build on **two physical Android phones**: speech
on one was recognised, sent as text, and spoken aloud on the other. Before that date the
Bluetooth link had never once completed a conversation on this project.

Screenshots and `adb logcat` from both devices — showing the CONNECTED state, packet sizes,
transport used, languages tested, and mouth-to-ear wall clock — are held with the team and
available on request. They are not in this repository because they were captured on
private handsets.
