---
topic: history.find
title: История: найти прогон и его файлы
summary: history list/show/import, где живёт SQLite, как читать статус файлов и почему чтение ничего не создаёт.
related: history.costs, search.lexical, asr.transcribe, start.quick
---

## Где живёт история

Одна локальная база SQLite в `VOICEOVER_HOME` (или в платформенном data-каталоге пользователя). История **не** привязана к текущему рабочему каталогу: команды `voiceover history ...` работают из любого места.

```bash
voiceover history list --json
voiceover history list --operation tts --status completed --limit 20 --json
voiceover history show 11111111-1111-1111-1111-111111111111 --json   # UUID или точная метка
```

`history show` принимает внутренний UUID, а UUID-подобное значение, не совпавшее с прогоном, пробуется как точная пользовательская метка.

## Импорт старых прогонов

```bash
voiceover history import ./out --dry-run --json
voiceover history import ./out --json
```

Импорт читает старые деревья `out/<run-id>`, транзакционен и идемпотентен, исходные файлы не трогает. `--dry-run` ничего не пишет и честно показывает `discovered`, `importable`, `already_imported`, `missing_text`, `missing_audio`.

## Гарантии чтения

- Пустая или отсутствующая база — это пустая история: ничего не создаётся, ни home, ни файл базы.
- Чтение идёт через read-only-шов: без миграций, DDL, переключения WAL и без создания sidecar-файлов.
- Повреждённая, чужая или слишком новая база fail-closed с фиксированной ошибкой, а не «неизвестный UUID».
- Симлинк на файл базы или каталог истории отклоняется до подключения.
- Публичный вывод не печатает сохранённые волатильные значения (подписанные URL, токены, произвольный текст) и не сериализует `config_snapshot`; текст источников и частей сообщается только булевой полнотой.

## Файлы прогона

`history show --json` и `search --json` сообщают путь и доступность аудио-артефакта, поэтому отсутствующий файл виден как статус доступности, а не как молчаливая поломка.

Поиск по содержимому сохранённых текстов — отдельная тема: `voiceover help search.lexical`. Деньги — `voiceover help history.costs`.
