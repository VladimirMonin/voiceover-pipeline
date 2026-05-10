# Gemini TTS prompting guide

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: режиссура Gemini 3.1 Flash TTS Preview, эмоции, audio tags, голоса, лимиты, chunking и production-пайплайн.

## Для чего этот guide

Используй этот документ, когда пользователь работает с `google/gemini-3.1-flash-tts-preview`,
`format: gemini-dialogue`, подкастами, диалогами, эмоциями, voice direction,
inline audio tags или длинной озвучкой. Главная идея: Gemini TTS — не обычный
«робот-чтец», а LLM-TTS исполнитель, которому нужна понятная режиссура.

## Источники и ориентиры

| Источник | Для чего полезен |
|---|---|
| Google AI Developers: Speech generation | Audio tags, advanced prompting, ограничения |
| Google AI Developers: model page | Token limits, preview status, model capabilities |
| Google Cloud Text-to-Speech Gemini-TTS | Byte limits, prompt/text split, output duration caveats |
| Google Cloud Blog | Expressive tags and best practices |
| LiveKit guide | Community prompt structure and common mistakes |
| GoogleCloudPlatform Voice Director Skill | Persona, scene, performance, transcript structure |

## Model facts

| Parameter | Value |
|---|---|
| Model id | `google/gemini-3.1-flash-tts-preview` |
| Input | Text |
| Output | Audio |
| Input token limit | 8192 tokens |
| Output token limit | 16384 tokens |
| Batch API | supported |
| Caching | not supported |
| Live API | not supported |
| Structured output | not supported |
| Status | Preview |

Treat the model as strong for prepared, expressive, directed voiceover. Do not
expect realtime streaming behavior or infinite monolithic generation.

## Why prompt it differently

Classic TTS mostly receives text plus fixed parameters. Gemini TTS interprets the
whole prompt as context. That is powerful, but risky: if service directions and
spoken text are mixed, the model can read director notes aloud. Always separate
direction from transcript.

## Canonical prompt skeleton

```text
Synthesize speech for the performance defined below.
The profile, scene, performance notes, and context are direction only.
Do NOT speak them.
Speak ONLY the lines under #### TRANSCRIPT.

# AUDIO PROFILE: Host A
## "Warm technical podcaster"

## SCENE: Evening recording session
A calm studio environment. The host is focused, warm, and speaking to an attentive audience.

### PERFORMANCE
Style: Thoughtful, warm, conversational, emotionally alive.
Pace: Natural podcast pace with small pauses after important ideas.
Accent: Natural Russian speech.

### CONTEXT
This is an educational podcast. The speaker should sound consistent, composed, and human.

#### TRANSCRIPT
[thoughtfully] Сегодня разберём, как правильно промптить Gemini TTS, [short pause] чтобы голос звучал выразительно.
```

| Block | Purpose |
|---|---|
| `AUDIO PROFILE` | persona, role, archetype, voice identity |
| `SCENE` | physical and emotional environment |
| `PERFORMANCE` | style, pace, accent, delivery, emotional baseline |
| `CONTEXT` | why this speech exists and who hears it |
| `TRANSCRIPT` | exact words to speak |

## Language policy

| Element | Recommended language |
|---|---|
| `AUDIO PROFILE`, `SCENE`, `PERFORMANCE`, `CONTEXT` | English |
| Audio tags | English |
| Transcript | Russian or other target language |
| User-facing UI controls | Any language, then map internally |

Russian transcript is fine. For production stability, keep service directions and
tags in English. Avoid production tags like `[плачет]`; map them to English tags
and style text.

## Audio tags

Audio tags are local delivery modifiers inside square brackets. They are best for
local changes: a pause, sigh, laugh, whisper, shout, gasp, or short shift in tone.
They should not replace the overall `PERFORMANCE` direction.

| Category | Safe starter tags |
|---|---|
| Soft delivery | `[warmly]`, `[thoughtfully]`, `[gently]`, `[calmly]` |
| Reactions | `[sighs]`, `[laughs]`, `[giggles]`, `[gasp]`, `[cough]`, `[uhm]` |
| Strong local shifts | `[whispers]`, `[shouting]` |
| Emotions | `[crying]`, `[serious]`, `[panicked]`, `[trembling]`, `[tired]`, `[amazed]`, `[curious]`, `[excited]` |
| Pauses | `[short pause]`, `[medium pause]`, `[long pause]` |

## Bracketed tag behavior

| Mode | Meaning | Examples | Reliability |
|---|---|---|---|
| Non-speech sounds | tag becomes a sound, tag not spoken | `[sighs]`, `[laughs]`, `[gasp]`, `[uhm]` | high |
| Style modifiers | tag changes following delivery | `[whispers]`, `[shouting]`, `[sarcastic]` | high |
| Vocalized markup | tag may be spoken as a word | unusual adjective tags | risky |

Production rule: keep the allowed tag dictionary small. Expand only after short
test generations with `--limit-chunks 2`.

## Emotion recipes

### General emotion belongs in `PERFORMANCE`

Bad:

```text
[sad] Я думал, что уже спокойно могу об этом говорить.
```

Better:

```text
### PERFORMANCE
Style: Sad but controlled. The voice carries emotion, but the speaker tries to stay composed.
Pace: Slow, with small emotional hesitations.

#### TRANSCRIPT
[sighs] Я думал, что уже спокойно могу об этом говорить, [short pause] но, кажется, пока не могу.
```

### Tearful but controlled

```text
### PERFORMANCE
Style: Tearful but controlled. The voice is fragile, sincere, and emotionally restrained.
Pace: Slow, with small hesitations and breathing space.

#### TRANSCRIPT
[crying] Я думал, что уже спокойно могу об этом говорить, [short pause] но, кажется, до конца всё ещё не могу. [sighs]
```

### Laughing through tears

```text
### PERFORMANCE
Style: The speaker is tearful but suddenly finds a small, fragile laugh inside the sadness. The laugh should feel human and involuntary, not cheerful.
Pace: Uneven, with small pauses and a soft emotional break.

#### TRANSCRIPT
[crying] Самое странное, что я всё равно улыбаюсь, [laughs] хотя вообще-то это совсем не смешно... [sighs]
```

### Cough or reset

```text
### PERFORMANCE
Style: Slightly nervous but trying to sound professional.
Pace: Measured, with a small reset before the key sentence.

#### TRANSCRIPT
[cough] Простите, [short pause] давайте я сформулирую это точнее.
```

## Voice choice matters

Pick the voice first, then write the profile around it. Do not force a bright,
youthful voice into a gravelly detective role.

| Task | Candidate voices | Why |
|---|---|---|
| Warm educational podcast | `Sulafat`, `Achernar`, `Zephyr` | warm, pleasant delivery |
| Clear technical narrator | `Kore`, `Charon`, `Rasalgethi` | firm, informative tone |
| Youthful or bright character | `Leda`, `Puck`, `Zephyr` | youthful, upbeat, bright |
| Gravelly late-night tone | `Algenib` | gravelly profile |
| Mature narrator | `Gacrux`, `Schedar` | mature, even tone |

For hoarse voice, prefer a compatible voice plus `Style: low, raspy, dry, intimate`.
For youthful voice, prefer a compatible voice plus `Style: youthful, curious,
bright, simple, sincere`. Do not rely on `[child voice]`.

## Emotion presets for apps

| UI emotion | Style prompt | Transcript tags |
|---|---|---|
| Warm | `Warm, sincere, conversational` | `[warmly]`, `[thoughtfully]` |
| Reflective | `Thoughtful, reflective, with small pauses` | `[thoughtfully]`, `[short pause]` |
| Sad | `Sad but controlled, voice carries feeling` | `[sighs]`, `[short pause]` |
| Crying | `Tearful but controlled, fragile voice` | `[crying]`, `[sighs]` |
| Amused | `Amused, natural, not exaggerated` | `[laughs]`, `[giggles]` |
| Laughing through tears | `Tearful with a fragile involuntary laugh` | `[crying]`, `[laughs]`, `[sighs]` |
| Nervous | `Nervous but trying to stay composed` | `[short pause]`, `[uhm]` |
| Tense | `Tense, alert, slightly breathless` | `[gasp]`, `[short pause]` |
| Raspy | `Low, raspy, gravelly, intimate` | prefer voice choice, minimal tags |
| Youthful | `Youthful, bright, curious, simple` | prefer voice choice, minimal tags |

## Limits and chunking

Formal limits are not production targets. Long generations can drift in voice,
prosody, and emotional consistency.

| Limit or guideline | Value |
|---|---|
| Gemini API input token limit | 8192 tokens |
| Gemini API output token limit | 16384 tokens |
| Cloud TTS `text` field | about 4000 bytes |
| Cloud TTS `prompt` field | about 4000 bytes |
| Cloud TTS prompt + text | about 8000 bytes |
| Output audio caveat | about 655 seconds can be truncated |
| Practical chunk duration | 90-180 seconds |
| Upper practical ceiling | 3-4 minutes, not default |
| Russian podcast chunk | 250-450 words |
| Emotional Russian chunk | 180-350 words |

Russian UTF-8 text uses more bytes than ASCII. Keep direction compact, especially
when the transcript is Cyrillic.

## Production chunking flow

1. Split a long podcast or chapter into semantic scenes.
2. Estimate each block duration.
3. Split blocks longer than about 180 seconds.
4. Add local scene/style only where needed.
5. Insert inline tags only at actual delivery changes.
6. Generate a short test: `--limit-chunks 2`.
7. Generate production audio with `--resume`, never default `--overwrite`.
8. Use `status` and `run_state.json` to inspect partial runs.
9. Use `concat --format ogg` for partial review files.
10. Run `timings` separately after audio is stable.

## Advanced techniques

- Concrete scene beats abstract role. Prefer `Late evening studio...` over `friendly and emotional`.
- Avoid deadening words: `flat`, `monotone`, `quiet`, `careful`, `no rush` can make delivery stiff.
- Write natural transcript punctuation. Too many short sentences can sound robotic.
- Do not repeat exact transcript words inside direction notes; describe the effect instead.
- Keep service direction compact and consistent across chunks to preserve character identity.

## Templates and QA

For prompt templates, project-native examples, QA checklist and anti-patterns,
read `docs/12-gemini-prompting-templates.md`.
