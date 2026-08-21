---
format: gemini-dialogue
language: ru
model: google/gemini-3.1-flash-tts-preview
speakers:
  Host:
    display_name: Ведущая
    voice: Kore
    profile: warm, confident technical host
  Guest:
    display_name: Гость
    voice: Puck
    profile: calm, thoughtful technical expert
vibe: >
  Russian technical podcast. Natural question-and-answer conversation.
allowed_tags:
  - warmly
  - curious
  - serious
  - short pause
max_chunk_bytes: 3500
---

Host: [warmly] Что умеет утилита?
Guest: Она создаёт озвучку, тайминги и субтитры.

******

Host: [curious] Можно работать полностью локально?
Guest: Да. Для одноголосой озвучки доступны OmniVoice и Qwen.
