"""Local SQLite vector database for warehouse traffic rules.

The project keeps the source documents in ``warehouse_rules.json`` and builds a
small persistent SQLite index next to it.  Embeddings are deterministic hashed
bag-of-words vectors.  When PyTorch is installed, vectorization and cosine
search use torch tensors; otherwise the module falls back to pure Python so the
Webots scene remains runnable on a minimal classroom machine.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    # PyTorch делает векторизацию и batch cosine search ближе к реальному ML pipeline.
    import torch
except ImportError:  # pragma: no cover - fallback нужен для машин без PyTorch.
    # None означает, что ниже будет использована чистая Python-реализация.
    torch = None


# DIMENSION — размерность локального embedding-пространства.
DIMENSION = 192
# TOKEN_RE выделяет английские/русские слова, числа и идентификаторы ребер.
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-я_]+", re.UNICODE)
# ENCODER_NAME попадает в SQLite metadata, чтобы понимать, чем построен индекс.
ENCODER_NAME = "torch_hashing_bow_v1"


# DEFAULT_DOCUMENTS используются, если warehouse_rules.json отсутствует.
DEFAULT_DOCUMENTS = [
    {
        "id": "traffic_right_hand",
        "text": (
            "At aisle intersections robots follow deadline priority first; "
            "if deadlines are equal, the robot that has another robot on its "
            "right-hand side yields. Loaded robots keep priority over empty "
            "robots when the deadline class is equal."
        ),
    },
    {
        "id": "collision_static",
        "text": (
            "A route segment is valid only if the robot safety disk does not "
            "intersect racks, chargers, walls or closed aisles. If a segment "
            "is blocked, the operational layer replans with A star on the "
            "warehouse occupancy grid."
        ),
    },
    {
        "id": "collision_dynamic",
        "text": (
            "Dynamic collision avoidance treats other robots as moving safety "
            "disks. A robot slows down or waits when the next short motion "
            "would reduce separation below the safety margin."
        ),
    },
    {
        "id": "three_level_control",
        "text": (
            "The three-level control model separates strategic policy, "
            "tactical coordination and operational execution. The LLM sets "
            "strategic weights, game-theoretic models assign tasks and routes, "
            "and local planners execute collision-free motion."
        ),
    },
    {
        "id": "congestion_policy",
        "text": (
            "The central crossing is short but shared. When several assigned "
            "routes use the same center edges, the congestion-game penalty "
            "should increase so some robots choose north, south or east bypasses."
        ),
    },
]


def _tokenize(text: str) -> List[str]:
    # Приводим текст к нижнему регистру, чтобы поиск не зависел от регистра.
    return TOKEN_RE.findall(text.lower())


def _stable_bucket(token: str, dimension: int) -> Tuple[int, float]:
    # Python hash рандомизируется между процессами, поэтому используем sha256.
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Первые 8 байт задают индекс корзины.
    index = int.from_bytes(digest[:8], "little") % dimension
    # Следующий байт задает знак, уменьшая коллизии hashing trick.
    sign = 1.0 if digest[8] % 2 == 0 else -1.0
    return index, sign


def _embed_python(text: str, dimension: int = DIMENSION) -> List[float]:
    # Создаем плотный вектор фиксированной размерности.
    vector = [0.0] * dimension
    # Каждый токен добавляет signed weight в стабильную hash-корзину.
    for token in _tokenize(text):
        index, sign = _stable_bucket(token, dimension)
        vector[index] += sign
    # Нормализация нужна, чтобы dot product был cosine similarity.
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        return vector
    return [value / norm for value in vector]


def _embed_tensor(text: str, dimension: int = DIMENSION):
    # Если torch отсутствует, вызывающий код использует _embed_python.
    if torch is None:
        return None
    # CPU-тензор здесь достаточен: документов мало, а Webots должен оставаться легким.
    vector = torch.zeros(dimension, dtype=torch.float32)
    # Векторизация намеренно детерминированная, без обучаемой модели и сетевых вызовов.
    for token in _tokenize(text):
        index, sign = _stable_bucket(token, dimension)
        vector[index] += sign
    # torch.linalg.norm делает тот же L2-норм, но через PyTorch backend.
    norm = torch.linalg.norm(vector)
    # Пустой текст оставляем нулевым вектором.
    if float(norm) <= 1e-9:
        return vector
    # Нормализуем вектор для cosine search.
    return vector / norm


def embed(text: str, dimension: int = DIMENSION) -> List[float]:
    # Публичная функция сохраняет старый интерфейс: возвращает list[float].
    tensor = _embed_tensor(text, dimension)
    # Если PyTorch доступен, возвращаем результат tensor pipeline.
    if tensor is not None:
        return tensor.tolist()
    # Иначе используем совместимый pure Python fallback.
    return _embed_python(text, dimension)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Для нормализованных векторов скалярное произведение равно cosine similarity.
    return sum(left * right for left, right in zip(a, b))


def lexical_overlap(query_text: str, document_text: str) -> float:
    # Берем те же токены, что и embedding, чтобы lexical score был воспроизводимым.
    query_tokens = set(_tokenize(query_text))
    # Документ тоже приводится к множеству токенов.
    document_tokens = set(_tokenize(document_text))
    # Пустой запрос или пустой документ не должны влиять на итоговую оценку.
    if not query_tokens or not document_tokens:
        return 0.0
    # Sørensen-Dice score стабилен на коротких правилах и понятен для inspect-логов.
    return 2.0 * len(query_tokens & document_tokens) / (len(query_tokens) + len(document_tokens))


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    # array('f') хранит float32 компактнее JSON и без внешних зависимостей.
    return array.array("f", vector).tobytes()


def _blob_to_vector(blob: bytes) -> List[float]:
    # Восстанавливаем float32-вектор из SQLite BLOB.
    values = array.array("f")
    # frombytes наполняет array без ручного unpack-цикла.
    values.frombytes(blob)
    return list(values)


@dataclass(frozen=True)
class RetrievedRule:
    # rule_id — id документа из warehouse_rules.json.
    rule_id: str
    # score — cosine similarity между query и документом.
    score: float
    # text — полный текст правила, который попадет в prompt.
    text: str


class LocalVectorStore:
    def __init__(self, path: str, db_path: Optional[str] = None) -> None:
        # path указывает на JSON-файл с исходными документами.
        self.path = path
        # db_path указывает на SQLite vector DB; по умолчанию лежит рядом с JSON.
        self.db_path = db_path or os.path.splitext(path)[0] + ".sqlite"
        # Загружаем документы из JSON или fallback-набор.
        self.source_documents = self._load_documents()
        # Инициализируем SQLite-схему и синхронизируем индекс с JSON.
        self._ensure_database()
        self._sync_documents()
        # Читаем документы уже из SQLite, чтобы поиск шел по векторной СУБД.
        self.documents = self._load_indexed_documents()

    @classmethod
    def default(cls) -> "LocalVectorStore":
        # База правил лежит рядом с этим Python-файлом.
        here = os.path.dirname(os.path.abspath(__file__))
        # Возвращаем store с проектным warehouse_rules.json и warehouse_rules.sqlite.
        return cls(os.path.join(here, "warehouse_rules.json"))

    def _connect(self) -> sqlite3.Connection:
        # sqlite3 входит в стандартную библиотеку и не требует отдельного сервера.
        connection = sqlite3.connect(self.db_path)
        # Row factory позволяет читать значения по именам колонок.
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_database(self) -> None:
        # Создаем директорию для SQLite-файла, если ее еще нет.
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Открываем соединение на время миграции схемы.
        with self._connect() as connection:
            # metadata хранит параметры индекса и помогает объяснить устройство СУБД.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            # documents хранит исходный текст и embedding как BLOB.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    encoder TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            # Индекс по encoder/dimension полезен для inspect-запросов и переиндексации.
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_encoder ON documents(encoder, dimension)"
            )
            # Обновляем metadata при каждом старте, чтобы статус был актуальным.
            self._set_metadata(connection, "dimension", str(DIMENSION))
            self._set_metadata(connection, "encoder", ENCODER_NAME)
            self._set_metadata(connection, "torch_available", "1" if torch is not None else "0")
            self._set_metadata(connection, "source_path", self.path)

    def _set_metadata(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        # UPSERT делает запись metadata идемпотентной.
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _load_documents(self) -> List[dict]:
        # Если файл удален, учебный пример все равно работает на DEFAULT_DOCUMENTS.
        if not os.path.exists(self.path):
            return list(DEFAULT_DOCUMENTS)
        # Читаем JSON в UTF-8, потому что правила могут быть русскоязычными.
        with open(self.path, "r", encoding="utf-8") as file:
            data = json.load(file)
        # Поддерживаем оба формата: {"documents": [...]} и просто [...].
        documents = data.get("documents", data)
        # Невалидный формат заменяем fallback-набором.
        if not isinstance(documents, list):
            return list(DEFAULT_DOCUMENTS)
        # Оставляем только документы с id и text, приводя значения к строкам.
        return [
            {"id": str(item["id"]), "text": str(item["text"])}
            for item in documents
            if "id" in item and "text" in item
        ]

    def _sync_documents(self) -> None:
        # valid_ids нужен, чтобы удалить из SQLite документы, удаленные из JSON.
        valid_ids = {document["id"] for document in self.source_documents}
        # Открываем соединение на время синхронизации.
        with self._connect() as connection:
            # Удаляем stale-документы из индекса.
            for row in connection.execute("SELECT id FROM documents"):
                if str(row["id"]) not in valid_ids:
                    connection.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            # Добавляем или обновляем каждый документ.
            for document in self.source_documents:
                # Embedding считается PyTorch-веткой, если torch установлен.
                vector = embed(document["text"])
                # INSERT ... ON CONFLICT обновляет embedding при изменении текста или encoder.
                connection.execute(
                    """
                    INSERT INTO documents(id, text, embedding, dimension, encoder, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text = excluded.text,
                        embedding = excluded.embedding,
                        dimension = excluded.dimension,
                        encoder = excluded.encoder,
                        updated_at = excluded.updated_at
                    """,
                    (
                        document["id"],
                        document["text"],
                        _vector_to_blob(vector),
                        DIMENSION,
                        ENCODER_NAME,
                        time.time(),
                    ),
                )

    def _load_indexed_documents(self) -> List[dict]:
        # Читаем документы из SQLite в стабильном порядке id.
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text, embedding FROM documents ORDER BY id"
            ).fetchall()
        # Возвращаем структуру, совместимую со старой JSON-базой.
        return [
            {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "embedding": _blob_to_vector(row["embedding"]),
            }
            for row in rows
        ]

    def query(self, text: str, top_k: int = 3) -> List[RetrievedRule]:
        # Строим embedding для текущего состояния склада или ручного запроса.
        query_vector = embed(text)
        # Если torch доступен, используем batch dot product по матрице документов.
        if torch is not None and self.documents:
            query_tensor = torch.tensor(query_vector, dtype=torch.float32)
            matrix = torch.tensor(
                [document["embedding"] for document in self.documents],
                dtype=torch.float32,
            )
            scores = torch.mv(matrix, query_tensor).tolist()
            scored = [
                RetrievedRule(document["id"], float(score), document["text"])
                for document, score in zip(self.documents, scores)
            ]
        else:
            # Pure Python fallback сохраняет поведение без PyTorch.
            scored = [
                RetrievedRule(
                    document["id"],
                    cosine(query_vector, document["embedding"]),
                    document["text"],
                )
                for document in self.documents
            ]
        # Маленькая база правил выигрывает от гибридного ранжирования: dense/hash
        # vector score ловит близость контекста, а lexical overlap стабилизирует
        # выбор при коротких эксплуатационных формулировках.
        scored = [
            RetrievedRule(
                rule.rule_id,
                0.68 * rule.score + 0.32 * lexical_overlap(text, rule.text),
                rule.text,
            )
            for rule in scored
        ]
        # Сортируем по убыванию релевантности.
        scored.sort(key=lambda item: item.score, reverse=True)
        # Возвращаем top-k правил для prompt.
        return scored[:top_k]

    def status(self) -> dict:
        # status используется логами и inspect_vector_db.py.
        return {
            "path": self.path,
            "db_path": self.db_path,
            "backend": "sqlite+pytorch" if torch is not None else "sqlite+python",
            "encoder": ENCODER_NAME,
            "documents": len(self.documents),
            "dimension": DIMENSION,
            "ids": [document["id"] for document in self.documents],
        }

    def all_documents(self) -> List[dict]:
        # Возвращаем копию без embedding, чтобы CLI не печатал длинные векторы по умолчанию.
        return [
            {"id": document["id"], "text": document["text"]}
            for document in self.documents
        ]

    def all_vectors(self) -> List[dict]:
        # Отдельный метод нужен inspect_vector_db.py для просмотра содержимого СУБД.
        return [
            {
                "id": document["id"],
                "text": document["text"],
                "embedding": list(document["embedding"]),
            }
            for document in self.documents
        ]

    def log_status(self, prefix: str = "[vector-db]") -> None:
        # Получаем компактное описание базы.
        status = self.status()
        # Печатаем одну строку, удобную для консоли Webots.
        print(
            "%s path=%s db=%s backend=%s encoder=%s documents=%d dimension=%d ids=%s"
            % (
                prefix,
                status["path"],
                status["db_path"],
                status["backend"],
                status["encoder"],
                status["documents"],
                status["dimension"],
                ",".join(status["ids"]),
            )
        )

    def log_query(self, query_text: str, results: Sequence[RetrievedRule], prefix: str = "[vector-query]") -> None:
        # Убираем лишние пробелы из JSON-запроса.
        compact_query = " ".join(query_text.split())
        # Длинный query обрезаем, чтобы логи оставались читаемыми.
        if len(compact_query) > 160:
            compact_query = compact_query[:157] + "..."
        # Формируем compact top-k: rule_id:score.
        result_text = ", ".join("%s:%.2f" % (item.rule_id, item.score) for item in results)
        # json.dumps экранирует кавычки внутри query.
        print("%s query=%s top=%s" % (prefix, json.dumps(compact_query), result_text))


def rules_to_prompt(rules: Iterable[RetrievedRule]) -> str:
    # Форматируем retrieved rules как список для LLM prompt.
    return "\n".join(
        "- %s (score=%.2f): %s" % (rule.rule_id, rule.score, rule.text)
        for rule in rules
    )
