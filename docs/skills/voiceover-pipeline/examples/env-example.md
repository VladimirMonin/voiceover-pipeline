# .env.example — шаблон для API-ключей

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Создай этот файл в корне проекта пользователя.
> НИКОГДА не заполняй его реальными ключами.

## Файл для создания

Создай файл `.env.example` с безопасными mock placeholder-ами (не с реальными ключами):

```env
# Copy this file to .env and fill real keys.
# .env is searched in CWD and upwards through parent directories.
# Never commit .env.

OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
POLZA_API_KEY=pza_xxxxxxxxxxxxxxxx
```

## Что сказать пользователю

«Я создал файл `.env.example` с шаблоном. Скопируй его в `.env` и впиши свои API-ключи.
Больше я спрашивать не буду.

- Для Polza нужен `POLZA_API_KEY` (начинается с `pza_`)
- Для OpenRouter нужен `OPENROUTER_API_KEY` (начинается с `sk-or-v1-`)
- Для Qwen-local ключи не нужны

После этого я проверю что ключи видны через `voiceover doctor`.»

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
