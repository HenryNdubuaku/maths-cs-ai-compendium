#!/usr/bin/env python3
"""
translate.py — качественный переводчик Markdown-сборника на русский язык через LLM.

Особенности:
  * Провайдеры: OpenRouter (по умолчанию) или локальный LM Studio — оба через
    OpenAI-совместимый API, один код-путь.
  * Математика (LaTeX $...$ / $$...$$), блоки кода, инлайн-код, пути картинок и
    ссылки НЕ маскируются — модель видит их в контексте, но обязана сохранять
    посимвольно. После перевода это проверяется автоматически (extract-and-compare);
    при расхождении — повтор с корректирующей подсказкой.
  * Глоссарий терминов (translate/glossary.md) передаётся модели для единообразия.
  * Разбивка на чанки по границам абзацев, но НИКОГДА не внутри блока кода или
    display-математики.
  * Резюмируемость: состояние в .translate_state.json (по хэшу результата), повторный
    запуск пропускает уже переведённые и неизменённые файлы.
  * Защита исходников: по умолчанию перевод «на месте», поэтому скрипт откажется
    работать при незакоммиченных изменениях в git (снимите ограничение --allow-dirty).

Использование см. в translate/README.md. Быстрый старт:
    export OPENROUTER_API_KEY=sk-or-...
    python3 translate/translate.py --dry-run          # что будет переведено
    python3 translate/translate.py                    # перевести всё на месте
    python3 translate/translate.py --provider lmstudio  # локально через LM Studio
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Требуется пакет 'requests'. Установите:  pip install requests")


def load_dotenv(root: Path) -> None:
    """Мини-загрузчик .env (без зависимостей): KEY=VALUE, без перезаписи уже заданных env."""
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


# --------------------------------------------------------------------------- #
# Провайдеры и модели по умолчанию                                            #
# --------------------------------------------------------------------------- #
PROVIDERS = {
    # OpenRouter: широкий выбор моделей, оплата по токенам.
    # По умолчанию Gemini 2.5 Pro — сильный технический перевод на русский,
    # длинный контекст и хорошее удержание LaTeX. Для экономии переключитесь на
    # google/gemini-2.5-flash (примерно в 15–20x дешевле, чуть ниже качество),
    # для максимальной аккуратности терминологии — anthropic/claude-sonnet-4.5.
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-pro",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # LM Studio: локальный OpenAI-совместимый сервер (вкладка Developer → Start server).
    # Модель задаётся тем, что вы загрузили; --model переопределяет.
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
        "api_key_env": "LMSTUDIO_API_KEY",  # обычно не нужен; шлём 'lm-studio'
    },
}

TARGET_LANG_DEFAULT = "Russian"

# --------------------------------------------------------------------------- #
# Системный промпт                                                            #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a professional technical translator specialising in mathematics, computer \
science and machine learning. Translate the Markdown fragment below from English to \
{target_lang}.

ABSOLUTE RULES — follow every one:

1. Translate ONLY natural-language prose: paragraph text, list item text, headings, \
table cell text, blockquote text, and the descriptive ALT text inside image links \
(the part between `![` and `]`).

2. Preserve BYTE-FOR-BYTE, never translate, never reformat:
   - All LaTeX math: inline `$...$` and display `$$...$$`. Do not touch anything \
inside the dollar signs, including `\\text{{}}`, variable names and operators.
   - All code: fenced blocks (```...``` or ~~~...~~~) and inline code (`...`). Keep \
code, identifiers, comments and string literals exactly as-is.
   - All link/image TARGETS — the URL or path inside `(...)`. Translate the visible \
link/alt text, but keep the path/URL identical (e.g. `../images/vector_3d.svg`).
   - HTML tags, attributes, and entities.
   - Markdown structure: `#` heading levels, list markers, indentation, bold/italic \
markers, blank lines, and the number of items.

3. Do NOT add, remove, summarise, explain, or reorder content. One source paragraph → \
one translated paragraph. Do not add a preamble like "Here is the translation".

4. Terminology: use the glossary consistently. Prefer established Russian technical \
terms; keep a widely-used English term in parentheses on first mention only if it aids \
clarity, but do NOT invent new parentheticals everywhere.

5. Register: neutral, precise, textbook style ("вы"-neutral, avoid slang). Keep the \
author's explanatory, slightly conversational tone where present.

Return ONLY the translated Markdown fragment, nothing else.

{glossary_block}"""


# --------------------------------------------------------------------------- #
# Извлечение защищённых элементов для валидации                               #
# --------------------------------------------------------------------------- #
FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")
DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(?:\\.|[^$\\\n])+?\$")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
LINK_TARGET_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
BARE_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _fenced_blocks(text: str) -> list[str]:
    """Возвращает содержимое ограждённых блоков кода целиком (по ```/~~~)."""
    blocks, buf, fence = [], [], None
    for line in text.splitlines():
        m = FENCE_RE.match(line)
        if fence is None:
            if m:
                fence = m.group(2)[0]
                buf = [line]
        else:
            buf.append(line)
            if m and m.group(2)[0] == fence and not m.group(3).strip():
                blocks.append("\n".join(buf))
                fence, buf = None, []
    if buf:  # незакрытый блок — тоже фиксируем
        blocks.append("\n".join(buf))
    return blocks


def protected_tokens(text: str) -> dict[str, list[str]]:
    """Мультимножества защищённых элементов для сравнения оригинала и перевода."""
    fenced = _fenced_blocks(text)
    # Уберём код из текста, чтобы инлайн-регексы не цепляли содержимое кода.
    masked = text
    for b in fenced:
        masked = masked.replace(b, "\n")
    return {
        "fenced_code": sorted(fenced),
        "display_math": sorted(DISPLAY_MATH_RE.findall(masked)),
        "inline_math": sorted(INLINE_MATH_RE.findall(DISPLAY_MATH_RE.sub("", masked))),
        "inline_code": sorted(INLINE_CODE_RE.findall(masked)),
        "link_targets": sorted(LINK_TARGET_RE.findall(masked)),
        "bare_urls": sorted(BARE_URL_RE.findall(masked)),
    }


def diff_tokens(src: dict, dst: dict) -> list[str]:
    """Список расхождений между защищёнными элементами оригинала и перевода."""
    problems = []
    for key in src:
        s, d = src[key], dst.get(key, [])
        if s != d:
            from collections import Counter

            missing = list((Counter(s) - Counter(d)).elements())
            extra = list((Counter(d) - Counter(s)).elements())
            if missing:
                problems.append(f"{key}: пропали {missing[:4]}")
            if extra:
                problems.append(f"{key}: появились лишние {extra[:4]}")
    return problems


# --------------------------------------------------------------------------- #
# Разбивка на чанки (не рвём код и display-математику)                        #
# --------------------------------------------------------------------------- #
def split_chunks(text: str, max_chars: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    cur: list[str] = []          # текущий прозаический чанк
    code: list[str] = []         # текущий блок кода (изолируется в отдельный чанк)
    fence: str | None = None
    in_display = False
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append("".join(cur))
            cur, cur_len = [], 0

    for line in lines:
        stripped = line.strip()
        m = FENCE_RE.match(line.rstrip("\n"))

        if fence is None and m:
            # начало блока кода: закрываем прозаический чанк, копим код отдельно
            fence = m.group(2)[0]
            flush()
            code = [line]
            continue
        if fence is not None:
            code.append(line)
            if m and m.group(2)[0] == fence and not m.group(3).strip():
                # конец блока кода -> отдельный чанк (LLM его не трогает)
                chunks.append("".join(code))
                code, fence = [], None
            continue

        # display-математика на отдельных строках ($$ ... $$)
        if stripped == "$$":
            in_display = not in_display

        cur.append(line)
        cur_len += len(line)
        safe_boundary = not in_display and stripped == ""
        if safe_boundary and cur_len >= max_chars:
            flush()

    if code:            # незакрытый блок кода — сохраняем как есть
        chunks.append("".join(code))
    flush()
    return chunks if chunks else [text]


def has_translatable_prose(chunk: str) -> bool:
    """Есть ли в чанке текст для перевода вне кода/математики/ссылок."""
    t = chunk
    for b in _fenced_blocks(t):
        t = t.replace(b, " ")
    t = DISPLAY_MATH_RE.sub(" ", t)
    t = INLINE_MATH_RE.sub(" ", t)
    t = INLINE_CODE_RE.sub(" ", t)
    t = LINK_TARGET_RE.sub(lambda m: m.group(0).split("](")[0], t)  # оставим alt-текст
    t = BARE_URL_RE.sub(" ", t)
    return bool(re.search(r"[A-Za-z]{3,}", t))


# --------------------------------------------------------------------------- #
# Клиент LLM                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class LLMClient:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.2
    timeout: int = 300
    _session: requests.Session = field(default_factory=requests.Session)

    def chat(self, system: str, user: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter любит эти заголовки (необязательно):
        headers.setdefault("HTTP-Referer", "https://github.com/egorthinks/maths-cs-ai-compendium-ru")
        headers.setdefault("X-Title", "maths-cs-ai-compendium-ru translator")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_err = None
        for attempt in range(5):
            try:
                r = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep = min(2 ** attempt, 30)
                time.sleep(sleep)
        raise RuntimeError(f"Запрос к LLM не удался после повторов: {last_err}")


# --------------------------------------------------------------------------- #
# Перевод одного чанка с валидацией                                           #
# --------------------------------------------------------------------------- #
def translate_chunk(client: LLMClient, system: str, chunk: str, max_retries: int) -> tuple[str, list[str]]:
    if not has_translatable_prose(chunk):
        return chunk, []  # чистый код/математика — оставляем как есть

    src_tokens = protected_tokens(chunk)
    user = chunk
    problems: list[str] = []
    result = chunk
    for attempt in range(max_retries + 1):
        result = client.chat(system, user).strip("\n")
        # снимем случайную обёртку ```markdown ... ```
        result = _unwrap_code_fence(result)
        problems = diff_tokens(src_tokens, protected_tokens(result))
        if not problems:
            return _preserve_edges(chunk, result), []
        # корректирующая подсказка
        user = (
            chunk
            + "\n\n---\nВНИМАНИЕ: в предыдущем переводе нарушены неизменяемые элементы:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\nПереведи заново, сохранив ВСЕ формулы, код, пути и ссылки посимвольно."
        )
    return _preserve_edges(chunk, result), problems  # вернём лучшее с флагом проблем


def _unwrap_code_fence(text: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\n(.*)\n```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def _preserve_edges(src: str, dst: str) -> str:
    """Сохраняем ведущие/замыкающие переводы строк исходного чанка."""
    lead = len(src) - len(src.lstrip("\n"))
    trail = len(src) - len(src.rstrip("\n"))
    return "\n" * lead + dst.strip("\n") + "\n" * trail


# --------------------------------------------------------------------------- #
# Обработка файлов                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    files_done: int = 0
    files_skipped: int = 0
    chunks: int = 0
    flagged: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def process_file(path: Path, root: Path, client: LLMClient, system: str, args, state: dict, stats: Stats):
    rel = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8")
    cur_sha = _sha(text)
    if not args.force and state.get(rel, {}).get("result_sha") == cur_sha:
        with stats.lock:
            stats.files_skipped += 1
        print(f"  ⏭  {rel} (уже переведён)")
        return

    chunks = split_chunks(text, args.max_chars)

    if args.dry_run:
        # Без сетевых вызовов: только показываем план.
        prose = sum(1 for c in chunks if has_translatable_prose(c))
        print(f"  📝 {rel}: {len(chunks)} чанк(ов), из них с текстом для перевода: {prose}")
        with stats.lock:
            stats.files_done += 1
            stats.chunks += len(chunks)
        return

    out_parts: list[str] = [""] * len(chunks)
    file_problems: list[str] = []

    def work(i_chunk):
        i, chunk = i_chunk
        translated, problems = translate_chunk(client, system, chunk, args.max_retries)
        return i, translated, problems

    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, translated, problems in ex.map(work, enumerate(chunks)):
            out_parts[i] = translated
            if problems:
                file_problems.append(f"чанк {i}: " + "; ".join(problems))

    out_text = "".join(out_parts)
    dest = path if not args.output_dir else (Path(args.output_dir) / rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out_text, encoding="utf-8")
    state[rel] = {"result_sha": _sha(out_text), "model": client.model, "ts": int(time.time())}
    _save_state(root, state)
    flag = " ⚠ (см. отчёт)" if file_problems else ""
    print(f"  ✅ {rel}: {len(chunks)} чанк(ов){flag}")

    with stats.lock:
        stats.files_done += 1
        stats.chunks += len(chunks)
        if file_problems:
            stats.flagged.append(f"{rel}\n    " + "\n    ".join(file_problems))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_path(root: Path) -> Path:
    return root / ".translate_state.json"


def _load_state(root: Path) -> dict:
    p = _state_path(root)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


_STATE_LOCK = threading.Lock()


def _save_state(root: Path, state: dict):
    with _STATE_LOCK:
        _state_path(root).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Глоссарий                                                                    #
# --------------------------------------------------------------------------- #
def build_glossary_block(glossary_path: Path | None) -> str:
    if not glossary_path or not glossary_path.exists():
        return ""
    lines = []
    for line in glossary_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # формат таблицы:  | English | Русский | ...  или  English -> Русский
        if line.startswith("|") and "|" in line[1:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and cells[0].lower() not in ("english", "term", "---", ":---", ":---:"):
                if not set(cells[0]) <= set("-: "):
                    lines.append(f"{cells[0]} = {cells[1]}")
        elif "->" in line and not line.startswith("#"):
            en, ru = line.split("->", 1)
            lines.append(f"{en.strip()} = {ru.strip()}")
    if not lines:
        return ""
    return "GLOSSARY (English = Русский), use consistently:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Выбор файлов                                                                 #
# --------------------------------------------------------------------------- #
def collect_files(root: Path, patterns: list[str]) -> list[Path]:
    result: list[Path] = []
    seen = set()
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p.suffix in (".md", ".txt") and p not in seen and ".git" not in p.parts:
                seen.add(p)
                result.append(p)
    return result


# --------------------------------------------------------------------------- #
# git-защита                                                                   #
# --------------------------------------------------------------------------- #
def git_dirty(root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None  # не git-репозиторий
        # игнорируем сам state-файл и папку translate/
        dirty = [
            l for l in out.stdout.splitlines()
            if l[3:].strip() not in (".translate_state.json",)
            and not l[3:].strip().startswith("translate/")
        ]
        return bool(dirty)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    root_default = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Перевод Markdown-сборника на русский через LLM.")
    ap.add_argument("--provider", choices=PROVIDERS, default="openrouter")
    ap.add_argument("--model", default=None, help="переопределить модель провайдера")
    ap.add_argument("--base-url", default=None, help="переопределить базовый URL API")
    ap.add_argument("--api-key", default=None, help="ключ API (иначе из переменной окружения)")
    ap.add_argument("--target-lang", default=TARGET_LANG_DEFAULT)
    ap.add_argument("--root", default=str(root_default), help="корень репозитория")
    ap.add_argument(
        "--files", nargs="*",
        default=["chapter */*.md", "index.md", "README.md", "llms.txt"],
        help="glob-шаблоны относительно корня (по умолчанию все главы + index/README)",
    )
    ap.add_argument("--glossary", default=str(Path(__file__).resolve().parent / "glossary.md"))
    ap.add_argument("--output-dir", default=None, help="каталог вывода (по умолчанию — на месте)")
    ap.add_argument("--max-chars", type=int, default=6000, help="макс. размер чанка")
    ap.add_argument("--concurrency", type=int, default=4, help="параллельных запросов")
    ap.add_argument("--max-retries", type=int, default=2, help="повторов при нарушении формул/кода")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true", help="показать план без изменений")
    ap.add_argument("--force", action="store_true", help="перевести даже уже готовые файлы")
    ap.add_argument("--allow-dirty", action="store_true", help="не проверять чистоту git")
    ap.add_argument("--limit", type=int, default=0, help="перевести не более N файлов (для теста)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    load_dotenv(root)  # подхватим .env, если есть
    prov = PROVIDERS[args.provider]
    model = args.model or os.environ.get("TRANSLATE_MODEL") or prov["model"]
    base_url = args.base_url or prov["base_url"]
    api_key = args.api_key or os.environ.get(prov["api_key_env"], "")
    if args.provider == "lmstudio" and not api_key:
        api_key = "lm-studio"  # LM Studio игнорирует значение, но поле нужно
    if args.provider == "openrouter" and not api_key and not args.dry_run:
        sys.exit(f"Нет ключа. Задайте переменную окружения {prov['api_key_env']} или --api-key.")

    # git-защита (только при переводе на месте)
    if not args.dry_run and not args.output_dir and not args.allow_dirty:
        dirty = git_dirty(root)
        if dirty:
            sys.exit(
                "В git есть незакоммиченные изменения. Перевод идёт «на месте» и перезапишет\n"
                "английские файлы. Закоммитьте текущее состояние (тогда оригинал сохранится в\n"
                "истории git), затем запустите снова. Обойти проверку: --allow-dirty."
            )

    glossary_block = build_glossary_block(Path(args.glossary) if args.glossary else None)
    system = SYSTEM_PROMPT.format(target_lang=args.target_lang, glossary_block=glossary_block)

    files = collect_files(root, args.files)
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit("Не найдено ни одного .md по заданным шаблонам.")

    client = LLMClient(base_url=base_url, model=model, api_key=api_key, temperature=args.temperature)
    state = _load_state(root)
    stats = Stats()

    print(f"Провайдер: {args.provider}   Модель: {model}")
    print(f"Файлов к обработке: {len(files)}   Параллелизм: {args.concurrency}   Чанк: {args.max_chars}")
    if glossary_block:
        print(f"Глоссарий: {args.glossary} ({glossary_block.count(chr(10))} терминов)")
    print("-" * 60)

    for path in files:
        try:
            process_file(path, root, client, system, args, state, stats)
        except KeyboardInterrupt:
            print("\nПрервано пользователем. Прогресс сохранён — можно продолжить позже.")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {path.relative_to(root)}: {e}")

    print("-" * 60)
    print(f"Готово: {stats.files_done}   Пропущено: {stats.files_skipped}   Чанков: {stats.chunks}")
    if stats.flagged:
        report = root / "translate" / "translation_report.txt"
        report.write_text(
            "Файлы с расхождениями в формулах/коде/ссылках (проверьте вручную):\n\n"
            + "\n\n".join(stats.flagged),
            encoding="utf-8",
        )
        print(f"⚠  {len(stats.flagged)} файл(ов) с предупреждениями — подробности: {report}")


if __name__ == "__main__":
    main()
