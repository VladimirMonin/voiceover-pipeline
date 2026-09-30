# .env.example — шаблон для API-ключей

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Создай только этот шаблон в корне проекта пользователя.
> НИКОГДА не создавай, не копируй и не читай реальный `.env`.

## Файл для создания

Создай файл `.env.example` с безопасными mock placeholder-ами (не с реальными ключами):

```env
# This is a template with synthetic placeholders. Never commit a real env file.
# The user creates a private env file outside the agent tools, or uses process
# environment variables. There is no parent-directory .env search.

OPENROUTER_API_KEY=sk-or-...xxxx
POLZA_API_KEY=pza_xxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
X_AI_API_KEY=xai-xxxxxxxxxxxxxxxx
```

Не выполняй `cp .env.example .env` / `Copy-Item .env.example .env` и не создавай
`.env` сам: реальный env-файл пользователь создаёт вручную вне инструментов агента.

## Что сказать пользователю

«Я создал `.env.example` (шаблон с placeholder-ами). Создай приватный env-файл
вне инструментов агента и впиши туда свои ключи, либо задай переменные окружения
процесса. Больше я ключи спрашивать не буду.

- Для Polza нужен `POLZA_API_KEY` (начинается с `pza_`)
- Для OpenRouter нужен `OPENROUTER_API_KEY` (начинается с `sk-or-v1-`)
- Для Groq Whisper нужен `GROQ_API_KEY` (начинается с `gsk_`)
- Для xAI STT нужен `X_AI_API_KEY` (начинается с `xai-`)
- Для Qwen-local и Faster-Whisper ключи не нужны

После этого я проверю что ключи видны через `voiceover doctor` в одобренном
окружении. Порядок разрешения: непустое окружение процесса → явный
`voiceover --env-file PATH ...` → `<CWD>/.env`; поиска по родительским каталогам нет.»

## Где взять ключи

- **Polza:** https://polza.ai/ → личный кабинет → API ключи
- **OpenRouter:** https://openrouter.ai/keys → создать ключ

## Проверка .gitignore

Убедись что `.gitignore` содержит строку `.env`.
Если нет — добавь:

```gitignore
# Secrets
.env
```
