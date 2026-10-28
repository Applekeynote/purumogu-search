# Audio and footage

## Concept film — 2026-10-28

`purumogu-concept-film.mp4` is a separate 149.5-second film made from the user's supplied narration. Voice: **音読さん Beta / Gray**, high quality, 1.0x, pitch 0, generated voice seed −913328603. Audio SHA-256: `17d670fbd09e49269e1d4726da931e3a00802fd55b5951b1a3c8f1dae9c9ed04`.

Music: [Purumogu — Signals in the Void](https://suno.com/song/0c1e4f5d-82cc-4f8e-b5ab-2b1b2cb2c128), newly generated with Suno V6 on the user's existing Pro plan. No new subscription was purchased. It is looped with native Resolve crossfades.

The film's cutting, typography motion, Fusion compositing, fades, loudness normalization, mixing and final H.264/AAC export were performed in **DaVinci Resolve Studio 21.1**. The labeled mechanism illustration was rendered separately in Blender 5.1 / Cycles from the project's 7cm sphere/taper proportions. It is not a measured physical trial. Application shots are actual current Unity/Rust executions, captured with cinematic camera framing. Source resampling and static media wrapping do not synthesize motion or mix the final film.

Disaster photograph: [国土交通省・国道470号 能越自動車道 緊急復旧の状況](https://www.mlit.go.jp/road/r6noto/pdf/09.pdf), page 1, No.3 damaged-site photograph. [MLIT content reuse conditions](https://www.mlit.go.jp/link.html). Background context: [警察庁・2024年1月10日活動報告](https://www.npa.go.jp/bureau/security/biki/disaster/20240110.pdf). Attribution appears with the photograph and in the end credits. The film is independently produced; government authorship or endorsement is not implied.

The MIT software license does not relicense the narration, music or government photograph. Their respective provider/content terms remain applicable. Full production notes are in `concept_film_sources/README.md`; final frame/audio QC is in `../reports/concept-film-qc.json`.

## Earlier films

- Narration: [音読さん](https://ondoku3.com/), Beta / Ellis, voice seed 1961932066, high quality, speed 1.0, pitch 0. Phase 1 script: `narration.txt`. Rescue v2 script: `rescue-narration-v2.txt`, recording: `rescue-narration-v2.mp3` (118.12s, SHA-256 `1829e42621a0c17410a231618cb4b6ee2c5a44c31e6b06e4f564afa73dfda68b`). Style: calm explanation to a colleague, without an advertising delivery. Generated 2026-10-27.
- Music: **Purumogu - Small Traces**, made for this project using [Suno AI](https://suno.com/song/5e26875a-6fcd-47ff-9783-aa27df2dc668), V6. Downloaded on the user's existing Pro plan on 2026-10-27. No additional subscription was purchased. Original style prompt: quiet 84 BPM instrumental with felt piano, plucked bass and brushed percussion; no artist or song reference.
- Footage: the included Unity application, controlled by the included Rust service. Recorded from the application backbuffer. Title cards and measured charts were typeset in code. No generated simulation footage or stock rescue imagery is used.

The music is looped with a crossfade and mixed below the narration. The completed film includes the voice and music credits. The software's MIT license does **not** relicense these audio recordings. Their use remains subject to [Ondoku's terms](https://ondoku3.com/ja/post/terms-licence/) and [Suno's terms](https://suno.com/terms/).

Editing, chapter overlays, fades, loudness normalization, music crossfade, mixing and the final 1920 × 1080 / 30 fps MP4 export were performed in **DaVinci Resolve Studio 21.1**. The original Phase 1 film uses ITU-R BS.1770-4 narration normalization at −16 LUFS and music at −30 LUFS. The Rescue v2 timeline uses narration normalization at −18.5 LUFS and music at −32 LUFS, with crossfades and native Resolve clip gain. The Rescue v2 final export is still pending. Original backbuffer PNGs were converted to 30 fps source MP4s by their capture timestamps before ingest; this step uses no motion synthesis. Shot changes show separate executions and are labeled on screen. The portable edit plans are `resolve-edit-plan.json` for Phase 1 and `rescue-resolve-edit-plan.json` for Rescue v2.

Ondoku credit guidance: https://ondoku3.com/ja/post/about-credit/
Suno download policy: https://suno.com/blog/suno-updates-tos
