# Concept film sources — 2026-10-28

The supplied narration is preserved in `../ナレーション原稿_コンセプト映像_20261028.txt`.
Voice: Ondoku Beta, Gray, high quality, 1.0x, normal pitch. Generated from the entire supplied text, with documentary delivery instructions. Generated voice seed: -913328603.
The audio was downloaded through the user's authenticated Chrome session; no account credentials are stored here.

The final edit, typography animation, compositing, audio mixing, fades and export are performed in DaVinci Resolve Studio 21.1. Application captures remain actual Unity/Rust runs; cinematic camera motion changes only the recording viewpoint.
Mechanism illustrations, if included, are explicitly labeled as illustrations rather than measured physical trials. No disaster rescue performance or 95% accuracy is asserted by the concept film.

Primary context source: National Police Agency, 2024-01-10 activity report, https://www.npa.go.jp/bureau/security/biki/disaster/20240110.pdf — initial large-vehicle access was impaired by severed roads.
The actual disaster photograph used in the film is the original embedded `Image199.jpg` on page 1 of the Ministry of Land, Infrastructure, Transport and Tourism document [国道470号 能越自動車道 緊急復旧の状況](https://www.mlit.go.jp/road/r6noto/pdf/09.pdf). It depicts the No.3 damaged site and is retained as `mlit-85-Image199.jpg`. It is placed beside the 2024 Noto context text without inventing or animating damage. [MLIT reuse conditions](https://www.mlit.go.jp/link.html). MLIT is credited during the photograph and in the end credits. The GSI thumbnail inspected during research is not used in the final edit.

Music: [Purumogu — Signals in the Void](https://suno.com/song/0c1e4f5d-82cc-4f8e-b5ab-2b1b2cb2c128), V6, generated and downloaded through the user's existing Suno Pro account on 2026-10-28. Original prompt: instrumental 112 BPM dark cinematic electronica, precise sub bass, analog pulses, dry metallic clicks and sparse piano, controlled lift and a hopeful ending, with space for Japanese narration. No artist or track imitation. Native Resolve clip fades crossfade repetitions; this film does not claim that Suno produced one uninterrupted 149.5-second take.

## Recreate the source assets

1. Build the current application, then use `scripts/Record-Rescue.ps1`: E02 scene 1 / 36 nodes / world / 900 frames; E03 scene 2 / 24 nodes / world / 600 frames; H04 scene 13 / 36 nodes / map / 1000 frames; H04 / 48 nodes / world / 600 frames. Use the shot names in `../concept-film-edit-plan.json`, speed 1 and separate inference ports. The model is `models/swarm-v3-bootstrap/model.json`, not an accepted 95% model. Camera-only filming switches do not change the physical or learning model. Each capture saves actual wall-clock timestamps.
2. In Blender 5.1, run `scripts/create_concept_robot_stage.py -- --render` for the 720-frame labeled mechanism illustration. Cycles uses 24 samples with denoising. This is a dimensional illustration of the sphere and flattened tapers, not a recorded docking test. `--preview` creates one source preview. `PURUMOGU_EEVEE=1` is an optional faster preview renderer; the delivered animation was rendered with Cycles.
3. Run `scripts/prepare_concept_film_sources.py --artwork --hero` followed by the shot names. It typesets new graphics and resamples uncut backbuffer captures by their saved timestamps. It does not edit the film. Run `scripts/wrap_concept_stills.py --stage <ASCII-path>` to stage the sources and wrap static artwork in alpha-preserving, unanimated QuickTime Animation files. This avoids Resolve's image-insert duration limitation. Narration and the original Suno MP3 must already be present in `media/`.

Use the frame-level edit plan in Resolve at 30 fps. Video source in/out boundaries are exclusive, as checked against the actual SDK. Typography and captions are separate tracks; the scene comparison uses three additional video tracks. Twenty picture clips have native Fusion Transform motion; the two map clips add a restrained native SoftGlow. Narration is normalized to −17.5 LUFS in Resolve, score sections to −28 / −23 LUFS, with native clip fades. Render the full timeline as a **single clip**, 1080p30 H.264 + 48 kHz AAC. The delivered export used Resolve's NVIDIA encoder defaults: the SDK rejected the attempted explicit bitrate, so no fixed 6 Mbps is claimed.

The final local project is `../Purumogu_Concept_Film_20261028.drp`. It opens the single `Signals_in_the_Void_MASTER` timeline. The linked source media is currently staged in the ASCII `purumogu-film` directory in this task's visualization workspace. Copy that folder with the DRP when transferring the edit to another machine and relink the Media Pool there. Raw render frames and the 140 MB completed MP4 are omitted from Git.

## Concrete viewing checks

- 00:49: the product reveal uses the original 7cm proportions, with an explicit mechanism-illustration label.
- 01:09–01:17: cone-tip and spherical-face connections are shown, followed by separation.
- 01:31–01:51: actual observed points grow into the map; amber is an unconfirmed person cue, rose is relative gas reaction, and blue is observed water depth.

The benefit of this structure is that the concept can be understood before the detailed application interface. The limitation is that polished CG and simplified simulated disasters do not establish physical rescue performance. Retaining the separate actual captures and observed/inferred labels makes that distinction reviewable. Additional filming with real prototypes would be the stronger next validation step.
