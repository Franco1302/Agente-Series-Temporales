"""Benchmark de modelos Ollama para tool-calling y latencia.

Mide tres ejes para cada modelo:
  1. Latencia de respuesta simple (saludo)
  2. Latencia + acierto en tool calling (drift)
  3. Latencia + acierto en consultar_teoria

Ejecutar:
    PYTHONPATH=. .venv/bin/python -m scripts.bench_models
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.agent.prompts import build_system_prompt
from src.agent.tools import AGENT_TOOLS

OLLAMA_BASE_URL = "http://localhost:11434"

import sys

# Permite filtrar modelos: python -m scripts.bench_models qwen2.5:3b llama3.1:8b
DEFAULT_MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q4_K_M",
]
MODELS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS


@dataclass
class Trial:
    name: str
    user_msg: str
    expected_tool: str | None  # None = no se espera tool call


TRIALS: list[Trial] = [
    Trial("saludo", "hola", None),
    Trial(
        "drift",
        "Analiza el drift en 'data/temp_uploads/ventas.csv' sobre la columna 'precio'.",
        "detect_drift_kolmogorov_smirnov",
    ),
    Trial(
        "rag",
        "¿Qué es el test de Kolmogorov-Smirnov?",
        "consultar_teoria",
    ),
]


def _load_model(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    ).bind_tools(AGENT_TOOLS)


def _run_trial(llm, trial: Trial) -> tuple[float, str | None, int]:
    sys_prompt = build_system_prompt(csv_path=None, csv_metadata=None)
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=trial.user_msg),
    ]
    t0 = time.time()
    msg = llm.invoke(messages)
    dt = time.time() - t0

    tool_calls = getattr(msg, "tool_calls", None) or []
    tool_name: str | None = None
    if tool_calls:
        c = tool_calls[0]
        tool_name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)

    content_len = len(msg.content) if isinstance(msg.content, str) else 0
    return dt, tool_name, content_len


def main() -> None:
    print(f"{'modelo':<40} {'caso':<10} {'lat (s)':>10} {'tool':<35} {'ok':>4}")
    print("-" * 105)

    summary: dict[str, list[float]] = {}

    for model in MODELS:
        try:
            llm = _load_model(model)
        except Exception as exc:
            print(f"{model:<40} ERROR carga: {exc}")
            continue

        # Warm-up: cargar el modelo en VRAM
        try:
            _run_trial(llm, TRIALS[0])
        except Exception as exc:
            print(f"{model:<40} ERROR warm-up: {exc}")
            continue

        latencias = []
        for trial in TRIALS:
            try:
                dt, tool, _ = _run_trial(llm, trial)
            except Exception as exc:
                print(f"{model:<40} {trial.name:<10} ERROR: {exc}")
                continue

            if trial.expected_tool is None:
                ok = tool is None
            else:
                ok = tool == trial.expected_tool

            latencias.append(dt)
            print(
                f"{model:<40} {trial.name:<10} {dt:>10.2f} "
                f"{(tool or '(texto)'):<35} {'✓' if ok else '✗':>4}"
            )
        summary[model] = latencias
        print()

    print("\n=== RESUMEN ===")
    print(f"{'modelo':<40} {'promedio (s)':>14}")
    for model, lats in summary.items():
        if lats:
            print(f"{model:<40} {sum(lats)/len(lats):>14.2f}")


if __name__ == "__main__":
    main()
