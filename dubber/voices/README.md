# Native Hindi reference voices

Used only when `VOICE_ENGINE` is `hindi_then_seedvc`: OmniVoice speaks each line in one of these native Hindi voices
(so pronunciation is native), then Seed-VC changes the timbre to the original speaker's voice.

| File | Speaker | Source clip |
|---|---|---|
| `hi_male.wav` | male | FLEURS `hi_in` dev, `11206833549725473331.wav` |
| `hi_female.wav` | female | FLEURS `hi_in` dev, `14156493267883433925.wav` |

Source: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: leading/trailing silence trimmed, volume normalised,
saved as 16-bit WAV.

**To use your own voice instead:** replace a `.wav` with a clean 5–10 s recording of native Hindi speech (no music, no
echo) and put its exact words in `voices.json` → `text`.
