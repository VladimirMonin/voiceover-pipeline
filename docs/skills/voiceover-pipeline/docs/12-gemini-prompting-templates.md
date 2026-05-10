# Gemini prompting templates and QA

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: project-native examples, prompt templates, QA checklist и anti-patterns для Gemini TTS.

## Important project contract

The long prompt skeleton in `11-gemini-prompting.md` is conceptual. Do not paste
it wholesale into the spoken body of a `voiceover-pipeline` script.

Project-native mapping:

| Conceptual block | `format: voiceover` | `format: gemini-dialogue` |
|---|---|---|
| AUDIO PROFILE | `style_prompt` / `prompt` | `speakers.<alias>.profile` |
| SCENE | `style_prompt` / `prompt` | `vibe` or compact profile text |
| PERFORMANCE | `style_prompt` / `prompt` | `vibe`, `profile`, inline tags |
| CONTEXT | `style_prompt` / `prompt` | `vibe` |
| TRANSCRIPT | body text only | `Speaker1:` / `Speaker2:` lines only |

If the body contains `AUDIO PROFILE`, `SCENE`, `PERFORMANCE`, `CONTEXT` or
`#### TRANSCRIPT`, the user probably pasted the conceptual prompt into the wrong
place. Move direction into frontmatter and keep body as spoken transcript only.

## Project-native single-speaker example

```markdown
---
format: voiceover
provider: openrouter-tts
model: google/gemini-3.1-flash-tts-preview
voice: Puck
style_prompt: >
  Warm technical podcaster. Thoughtful, conversational, emotionally alive.
  Natural Russian speech. Direction is not spoken aloud.
max_chunk_chars: 2000
---

[thoughtfully] Сегодня разберём Gemini TTS, [short pause] и почему хороший голос начинается с режиссуры.
```

## Project-native dialogue example

```markdown
---
format: gemini-dialogue
language: ru
model: google/gemini-3.1-flash-tts-preview
speakers:
  Speaker1:
    display_name: Ведущий
    voice: Puck
    profile: calm, warm technical host
  Speaker2:
    display_name: Соведущая
    voice: Kore
    profile: curious, lively co-host, not exaggerated
vibe: >
  Friendly Russian technical podcast. Natural pace, thoughtful delivery,
  small pauses between speaker turns. Direction is not spoken aloud.
allowed_tags:
  - thoughtfully
  - curious
  - short pause
  - laughs
max_chunk_bytes: 3500
---

Speaker1: [thoughtfully] Сначала кажется, что TTS — это просто чтение текста.
Speaker2: [curious] Но модель ещё и интерпретирует режиссуру, верно?
Speaker1: Именно. [short pause] Поэтому структура промпта почти так же важна, как сам текст.
```

## Educational podcast template

```text
Synthesize speech for the performance defined below.
The profile, scene, performance notes, and context are direction only.
Do NOT speak them.
Speak ONLY the lines under #### TRANSCRIPT.

# AUDIO PROFILE: Host A
## "Warm educational podcaster"

## SCENE: Evening podcast studio
A calm, well-treated studio. The host speaks to an audience that wants clear explanations.

### PERFORMANCE
Style: Warm, precise, thoughtful, emotionally alive, but not theatrical.
Pace: Natural and steady, with small pauses after important points.
Accent: Natural Russian speech.

#### TRANSCRIPT
[thoughtfully] Сегодня речь пойдёт о Gemini TTS, [short pause] и почему результат начинается не с тегов, а со сцены.
```

Use this template as direction source, then map it into `style_prompt` or
`gemini-dialogue` frontmatter.

## Emotional line template

```text
### PERFORMANCE
Style: Tearful but controlled. The speaker is trying to stay composed.
Pace: Slow, with small hesitations and breathing space.

#### TRANSCRIPT
[crying] Я думал, что уже спокойно могу об этом говорить, [short pause] но, кажется, нет... [sighs]
```

## QA checklist

| Check | Question |
|---|---|
| Preamble | Does conceptual prompt say not to speak direction blocks? |
| Project mapping | Are directions in frontmatter, not body? |
| Transcript boundary | Is body spoken text only? |
| Voice fit | Does selected voice match the role? |
| Scene | Is there concrete context? |
| Performance | Is global emotion in style, not only tags? |
| Tags | Are tags English and from the allowed set? |
| Punctuation | Does text sound conversational? |
| Chunk size | Is it near 90-180 seconds / 250-450 Russian words? |
| Recovery | Are `--resume`, state, log, retry available? |

## Anti-patterns

- Pasting the full conceptual prompt into the spoken body.
- One huge podcast as a single request.
- Free-form Russian bracket tags in production.
- `sound emotional` without scene, motivation, or style.
- Trying to solve global emotion with one tag.
- Choosing a voice that fights the role.
- Overloading prompt with contradictory notes.
- Using `--overwrite` on paid chunks instead of `--resume`.
