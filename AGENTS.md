# AGENTS.md — voiceover-pipeline

> Авто-загружаемый контекст для AI-агентов.

## Быстрые команды

```bash
uv run voiceover doctor --json                   # проверка окружения
uv run voiceover list timing-providers --json    # 4 провайдера распознавания
uv run pytest tests/ -x --tb=short               # тесты (>120)
```

## Публикация в PyPI

```bash
# 1. Токен лежит в .env: PYPI_TOKEN=pypi-...
# 2. Версия уже в pyproject.toml
# 3. Собрать и опубликовать:
uv build
export $(grep -v '^#' .env | grep PYPI_TOKEN | xargs)
uv publish --token "$PYPI_TOKEN"

# 4. Тег:
git tag v$(grep version pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
git push --tags
```

## Где что лежит

| Что | Где |
|-----|-----|
| Код провайдеров TTS | `src/voiceover_pipeline/providers/` |
| Код провайдеров STT | `src/voiceover_pipeline/providers/` (groq_whisper.py, xai_stt.py, openrouter_whisper.py) |
| CLI | `src/voiceover_pipeline/cli.py` |
| Конфиг | `src/voiceover_pipeline/config.py` |
| Документация навыка | `docs/skills/voiceover-pipeline/` (SKILL.md + docs/*.md) |
| OpenCode навык (Obsidian) | `E:\AUTO_OBSIDIAN\.opencode\skills\voiceover-pipeline\` |
| `.env` | Корень проекта. НИКОГДА НЕ ЧИТАТЬ содержимое. Токен PYPI_TOKEN внутри. |

## Провайдеры распознавания (timing)

| ID | Тип | Таймкоды | Ключ |
|----|-----|----------|------|
| `faster-whisper` | local | segment+word | — |
| `openrouter-whisper` | cloud | ❌ text only | `OPENROUTER_API_KEY` |
| `groq-whisper` | cloud | segment+word | `GROQ_API_KEY` |
| `xai-stt` | cloud | word+confidence | `X_AI_API_KEY` |

`openrouter-whisper` заблокирован для `timings`/`--with-timings` (exit code 40).

## CI/CD

Скрипт релиза: `scripts/release.ps1` (PowerShell, Windows).
На Linux: ручной bump + build + publish + tag (см. выше).
