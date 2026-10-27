# Audio and footage

- Narration: [音読さん](https://ondoku3.com/), Beta / Ellis, voice seed 1961932066, high quality, speed 1.0, pitch 0. Phase 1 script: `narration.txt`. Rescue v2 script: `rescue-narration-v2.txt`, recording: `rescue-narration-v2.mp3` (118.12s, SHA-256 `1829e42621a0c17410a231618cb4b6ee2c5a44c31e6b06e4f564afa73dfda68b`). Style: calm explanation to a colleague, without an advertising delivery. Generated 2026-10-27.
- Music: **Purumogu - Small Traces**, made for this project using [Suno AI](https://suno.com/song/5e26875a-6fcd-47ff-9783-aa27df2dc668), V6. Downloaded on the user's existing Pro plan on 2026-10-27. No additional subscription was purchased. Original style prompt: quiet 84 BPM instrumental with felt piano, plucked bass and brushed percussion; no artist or song reference.
- Footage: the included Unity application, controlled by the included Rust service. Recorded from the application backbuffer. Title cards and measured charts were typeset in code. No generated simulation footage or stock rescue imagery is used.

The music is looped with a crossfade and mixed below the narration. The completed film includes the voice and music credits. The software's MIT license does **not** relicense these audio recordings. Their use remains subject to [Ondoku's terms](https://ondoku3.com/ja/post/terms-licence/) and [Suno's terms](https://suno.com/terms/).

Editing, chapter overlays, fades, loudness normalization, music crossfade, mixing and the final 1920 × 1080 / 30 fps MP4 export were performed in **DaVinci Resolve Studio 21.1**. The original Phase 1 film uses ITU-R BS.1770-4 narration normalization at −16 LUFS and music at −30 LUFS. The Rescue v2 timeline uses narration normalization at −18.5 LUFS and music at −32 LUFS, with crossfades and native Resolve clip gain. The Rescue v2 final export is still pending. Original backbuffer PNGs were converted to 30 fps source MP4s by their capture timestamps before ingest; this step uses no motion synthesis. Shot changes show separate executions and are labeled on screen. The portable edit plan is `resolve-edit-plan.json`.

Ondoku credit guidance: https://ondoku3.com/ja/post/about-credit/
Suno download policy: https://suno.com/blog/suno-updates-tos
