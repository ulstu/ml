"""Inspect the local vector database used by the warehouse LLM pipeline."""

from __future__ import annotations

import argparse
import json

from vector_memory import LocalVectorStore


def main() -> int:
    # argparse делает утилиту удобной для запуска из терминала.
    parser = argparse.ArgumentParser(description="Inspect warehouse_rules.json and run local vector search.")
    # --query включает режим similarity search.
    parser.add_argument("--query", default="", help="Optional query text for similarity search.")
    # --top-k управляет числом результатов поиска.
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved documents to print.")
    # --json полезен для автоматических проверок или внешних инструментов.
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    # --vectors показывает embedding-содержимое SQLite vector DB.
    parser.add_argument("--vectors", action="store_true", help="Print stored embedding vectors.")
    # parse_args читает аргументы командной строки.
    args = parser.parse_args()

    # Загружаем локальную векторную базу из warehouse_rules.json.
    store = LocalVectorStore.default()
    # status содержит путь, число документов, размерность и id документов.
    status = store.status()
    # documents — все исходные документы базы.
    documents = store.all_documents()
    # vectors — документы вместе с embedding; включается только по запросу.
    vectors = store.all_vectors() if args.vectors else []
    # results заполняется только если пользователь передал --query.
    results = store.query(args.query, top_k=args.top_k) if args.query else []

    # JSON-режим печатает структуру без человекочитаемого форматирования.
    if args.json:
        print(
            json.dumps(
                {
                    "status": status,
                    "documents": documents,
                    "vectors": vectors,
                    "query": args.query,
                    "results": [
                        {"id": item.rule_id, "score": item.score, "text": item.text}
                        for item in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    # Человекочитаемый режим начинается со статуса базы.
    print("[vector-db] path=%s" % status["path"])
    print("[vector-db] sqlite=%s" % status["db_path"])
    print("[vector-db] backend=%s encoder=%s" % (status["backend"], status["encoder"]))
    print("[vector-db] documents=%d dimension=%d" % (status["documents"], status["dimension"]))
    print("[vector-db] ids=%s" % ", ".join(status["ids"]))
    print()
    # Затем печатаем все документы с номерами.
    for index, document in enumerate(documents, start=1):
        print("[%02d] %s" % (index, document["id"]))
        print("     %s" % document["text"])
    # По запросу печатаем первые компоненты embedding, чтобы не перегружать терминал.
    if args.vectors:
        print()
        print("[vector-db] stored embeddings")
        for item in vectors:
            preview = ", ".join("%.3f" % value for value in item["embedding"][:12])
            print("  id=%s dim=%d head=[%s, ...]" % (item["id"], len(item["embedding"]), preview))
    # Если был query, печатаем результаты similarity search.
    if args.query:
        print()
        print("[vector-query] %s" % args.query)
        for item in results:
            print("  score=%.3f id=%s" % (item.score, item.rule_id))
            print("  %s" % item.text)
    # 0 означает успешное выполнение CLI.
    return 0


if __name__ == "__main__":
    # raise SystemExit передает код возврата main() в shell.
    raise SystemExit(main())
