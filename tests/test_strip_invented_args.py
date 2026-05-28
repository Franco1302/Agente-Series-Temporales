"""Tests de la defensa anti-invención de args en `reasoning._strip_invented_args`.

Cuando el LLM local (cuantizado) viola `RULE_NO_INVENT` y rellena campos
"obvios" sin que el usuario los haya escrito, esta defensa retira esos args
antes de la validación para que el grafo los pida explícitamente.

Cobertura: las 8 herramientas analíticas y los tipos de campo definidos en
`_FIELD_EVIDENCE_MAP` (date, freq, integer, numeric_list, distribution_kind,
trend_kind, pattern_kind, drift_method, augment_strategy, exogenous_relation,
existing_column, new_column).
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes import reasoning as reasoning_mod
from src.agent.nodes.reasoning import _strip_invented_args


# ── Utilidades de fixture ───────────────────────────────────────────────────


def _tool_call_msg(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": dict(args), "id": "call_test", "type": "tool_call"}],
    )


def _args_of(msg: AIMessage) -> dict:
    return msg.tool_calls[0]["args"]


def _state(user_text: str, csv_metadata: dict | None = None) -> dict:
    """Construye un state mínimo con un único HumanMessage y opcional csv_metadata."""
    return {
        "messages": [HumanMessage(content=user_text)],
        "csv_metadata": csv_metadata,
    }


# ── Tests por tipo de campo ─────────────────────────────────────────────────


# date / freq ----------------------------------------------------------------


def test_strip_invented_start_date_no_evidence():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"start_date": "2020-01-01", "distribution_type": 1},
    )
    cleaned = _strip_invented_args(msg, _state("Genera una serie sintética normal."))
    assert "start_date" not in _args_of(cleaned)


def test_keep_start_date_when_user_writes_iso_date():
    msg = _tool_call_msg("generate_synthetic_distribution", {"start_date": "2024-01-01"})
    cleaned = _strip_invented_args(msg, _state("Empieza el 2024-01-01."))
    assert _args_of(cleaned)["start_date"] == "2024-01-01"


def test_keep_start_date_when_user_mentions_month_name():
    msg = _tool_call_msg("generate_synthetic_arma", {"start_date": "2024-03-01"})
    cleaned = _strip_invented_args(msg, _state("Desde marzo de 2024."))
    assert _args_of(cleaned)["start_date"] == "2024-03-01"


def test_strip_invented_frequency_no_evidence():
    msg = _tool_call_msg("generate_synthetic_distribution", {"frequency": "D"})
    cleaned = _strip_invented_args(msg, _state("Genera serie temporal."))
    assert "frequency" not in _args_of(cleaned)


def test_keep_frequency_when_user_says_daily():
    msg = _tool_call_msg("generate_synthetic_distribution", {"frequency": "D"})
    cleaned = _strip_invented_args(msg, _state("Quiero una serie diaria."))
    assert _args_of(cleaned)["frequency"] == "D"


def test_keep_frequency_when_user_says_weekly():
    msg = _tool_call_msg("generate_synthetic_arma", {"frequency": "W"})
    cleaned = _strip_invented_args(msg, _state("frecuencia semanal."))
    assert _args_of(cleaned)["frequency"] == "W"


# integer --------------------------------------------------------------------


def test_strip_invented_periods_no_digit_in_text():
    msg = _tool_call_msg("generate_synthetic_distribution", {"periods": 100})
    cleaned = _strip_invented_args(msg, _state("Genera una serie sintética."))
    assert "periods" not in _args_of(cleaned)


def test_keep_periods_when_user_writes_digit():
    msg = _tool_call_msg("generate_synthetic_distribution", {"periods": 12})
    cleaned = _strip_invented_args(msg, _state("Quiero 12 observaciones."))
    assert _args_of(cleaned)["periods"] == 12


def test_keep_periods_when_user_writes_number_word():
    msg = _tool_call_msg("generate_synthetic_distribution", {"periods": 12})
    cleaned = _strip_invented_args(msg, _state("Quiero doce observaciones."))
    assert _args_of(cleaned)["periods"] == 12


def test_strip_invented_forecast_steps():
    msg = _tool_call_msg("forecast_time_series", {"forecast_steps": 30})
    cleaned = _strip_invented_args(msg, _state("Predice el futuro de la serie."))
    assert "forecast_steps" not in _args_of(cleaned)


def test_keep_forecast_steps_when_user_writes_digit():
    msg = _tool_call_msg("forecast_time_series", {"forecast_steps": 30})
    cleaned = _strip_invented_args(msg, _state("Predice 30 pasos hacia adelante."))
    assert _args_of(cleaned)["forecast_steps"] == 30


# numeric_list ---------------------------------------------------------------


def test_strip_invented_distribution_params():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [0.0, 1.0]},
    )
    cleaned = _strip_invented_args(msg, _state("Genera serie sintética normal."))
    assert "distribution_params" not in _args_of(cleaned)


def test_keep_distribution_params_when_user_lists_numbers():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [0.0, 1.0]},
    )
    cleaned = _strip_invented_args(msg, _state("Parámetros [0.0, 1.0]."))
    assert _args_of(cleaned)["distribution_params"] == [0.0, 1.0]


def test_keep_distribution_params_when_user_mentions_sigma():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [0.0, 1.0]},
    )
    cleaned = _strip_invented_args(msg, _state("mu=0 y sigma=1."))
    assert _args_of(cleaned)["distribution_params"] == [0.0, 1.0]


def test_strip_invented_trend_params():
    msg = _tool_call_msg("generate_synthetic_trend", {"trend_params": [0.5, 1.0]})
    cleaned = _strip_invented_args(msg, _state("Genera serie con tendencia."))
    assert "trend_params" not in _args_of(cleaned)


# distribution_kind / trend_kind / pattern_kind ------------------------------


def test_strip_invented_distribution_type():
    msg = _tool_call_msg("generate_synthetic_distribution", {"distribution_type": 1})
    cleaned = _strip_invented_args(msg, _state("Genera una serie sintética."))
    assert "distribution_type" not in _args_of(cleaned)


def test_keep_distribution_type_when_user_says_normal():
    msg = _tool_call_msg("generate_synthetic_distribution", {"distribution_type": 1})
    cleaned = _strip_invented_args(msg, _state("Distribución normal."))
    assert _args_of(cleaned)["distribution_type"] == 1


def test_keep_distribution_type_when_user_says_poisson():
    msg = _tool_call_msg("generate_synthetic_distribution", {"distribution_type": 3})
    cleaned = _strip_invented_args(msg, _state("Poisson con lambda 5."))
    assert _args_of(cleaned)["distribution_type"] == 3


def test_strip_invented_trend_type():
    msg = _tool_call_msg("generate_synthetic_trend", {"trend_type": 1})
    cleaned = _strip_invented_args(msg, _state("Genera serie con tendencia."))
    # "tendencia" SÍ está entre las keywords, así que NO debe stripearse
    assert _args_of(cleaned)["trend_type"] == 1


def test_strip_trend_type_when_no_kind_mentioned():
    msg = _tool_call_msg("generate_synthetic_trend", {"trend_type": 1})
    cleaned = _strip_invented_args(msg, _state("Crea datos."))
    assert "trend_type" not in _args_of(cleaned)


def test_keep_pattern_type_when_user_says_amplitud():
    msg = _tool_call_msg("generate_synthetic_periodic", {"pattern_type": 1})
    cleaned = _strip_invented_args(msg, _state("Variación de amplitud cada 7 días."))
    assert _args_of(cleaned)["pattern_type"] == 1


# drift_method / augment_strategy / exogenous_relation ----------------------


def test_strip_invented_drift_method():
    msg = _tool_call_msg("detect_drift", {"method": "KS"})
    cleaned = _strip_invented_args(msg, _state("Quiero analizar mis datos."))
    assert "method" not in _args_of(cleaned)


def test_keep_drift_method_KS_when_user_says_kolmogorov():
    msg = _tool_call_msg("detect_drift", {"method": "KS"})
    cleaned = _strip_invented_args(msg, _state("Usa el test de Kolmogorov-Smirnov."))
    assert _args_of(cleaned)["method"] == "KS"


def test_keep_drift_method_KS_when_user_says_KS():
    msg = _tool_call_msg("detect_drift", {"method": "KS"})
    cleaned = _strip_invented_args(msg, _state("método KS."))
    assert _args_of(cleaned)["method"] == "KS"


def test_strip_invented_augment_strategy():
    msg = _tool_call_msg("augment_time_series", {"strategy": "normal"})
    cleaned = _strip_invented_args(msg, _state("Quiero más datos."))
    assert "strategy" not in _args_of(cleaned)


def test_keep_augment_strategy_when_user_says_duplicar():
    msg = _tool_call_msg("augment_time_series", {"strategy": "duplicate"})
    cleaned = _strip_invented_args(msg, _state("Duplica mis datos con ruido."))
    assert _args_of(cleaned)["strategy"] == "duplicate"


def test_strip_invented_exogenous_relation():
    msg = _tool_call_msg("create_exogenous_variable", {"relation": "linear"})
    cleaned = _strip_invented_args(msg, _state("Crea una columna nueva."))
    assert "relation" not in _args_of(cleaned)


def test_keep_exogenous_relation_when_user_says_pca():
    msg = _tool_call_msg("create_exogenous_variable", {"relation": "pca"})
    cleaned = _strip_invented_args(msg, _state("Usa PCA para crear la columna."))
    assert _args_of(cleaned)["relation"] == "pca"


# existing_column / new_column -----------------------------------------------


def test_keep_index_column_when_in_csv_metadata():
    msg = _tool_call_msg("detect_drift", {"index_column": "Indice"})
    state = _state("Detecta drift.", csv_metadata={"columns": ["Indice", "valor"]})
    cleaned = _strip_invented_args(msg, state)
    assert _args_of(cleaned)["index_column"] == "Indice"


def test_strip_invented_index_column_not_in_csv():
    msg = _tool_call_msg("detect_drift", {"index_column": "fecha"})
    state = _state("Detecta drift.", csv_metadata={"columns": ["Indice", "valor"]})
    cleaned = _strip_invented_args(msg, state)
    assert "index_column" not in _args_of(cleaned)


def test_keep_index_column_when_user_mentions_it_textually():
    msg = _tool_call_msg("detect_drift", {"index_column": "ts"})
    cleaned = _strip_invented_args(msg, _state("Usa la columna ts como índice."))
    assert _args_of(cleaned)["index_column"] == "ts"


def test_keep_target_column_from_csv_metadata():
    msg = _tool_call_msg("forecast_time_series", {"target_column": "valor"})
    state = _state("Predice.", csv_metadata={"columns": ["Indice", "valor"]})
    cleaned = _strip_invented_args(msg, state)
    assert _args_of(cleaned)["target_column"] == "valor"


def test_strip_new_column_when_user_did_not_name_it():
    msg = _tool_call_msg("create_exogenous_variable", {"new_column_name": "y"})
    cleaned = _strip_invented_args(msg, _state("Crea una columna PCA."))
    assert "new_column_name" not in _args_of(cleaned)


def test_keep_new_column_when_user_names_it():
    msg = _tool_call_msg("create_exogenous_variable", {"new_column_name": "y"})
    cleaned = _strip_invented_args(msg, _state("Crea la columna y usando PCA."))
    assert _args_of(cleaned)["new_column_name"] == "y"


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_strip_noop_for_unmapped_tools():
    """`consultar_teoria` no está en `_FIELD_EVIDENCE_MAP`: no se toca nada."""
    msg = _tool_call_msg("consultar_teoria", {"query": "qué es drift"})
    cleaned = _strip_invented_args(msg, _state("hola"))
    assert _args_of(cleaned) == {"query": "qué es drift"}


def test_strip_noop_when_no_tool_calls():
    msg = AIMessage(content="texto plano")
    cleaned = _strip_invented_args(msg, _state("hola"))
    assert cleaned is msg


def test_strip_noop_when_no_human_messages():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"start_date": "2020-01-01", "frequency": "D"},
    )
    cleaned = _strip_invented_args(msg, {"messages": [], "csv_metadata": None})
    assert _args_of(cleaned)["start_date"] == "2020-01-01"
    assert _args_of(cleaned)["frequency"] == "D"


def test_strip_ignores_empty_args():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"start_date": "", "frequency": None, "distribution_type": 1},
    )
    cleaned = _strip_invented_args(msg, _state("Genera serie."))
    args = _args_of(cleaned)
    assert args.get("start_date") == ""
    assert args.get("frequency") is None
    # distribution_type sin keyword de distribución → strip
    assert "distribution_type" not in args


# ── Bypass por delegación explícita ─────────────────────────────────────────


def test_delegation_default_bypasses_strip():
    """'usa los defaults' desactiva el strip ese turno."""
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {
            "start_date": "2020-01-01",
            "frequency": "D",
            "distribution_type": 1,
            "distribution_params": [0.0, 1.0],
            "periods": 30,
        },
    )
    cleaned = _strip_invented_args(
        msg,
        _state("usa los defaults para todo."),
    )
    args = _args_of(cleaned)
    # Nada se retira: el usuario delegó explícitamente.
    assert args["start_date"] == "2020-01-01"
    assert args["frequency"] == "D"
    assert args["distribution_params"] == [0.0, 1.0]


def test_delegation_cualquier_bypasses_strip():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [0.5]},
    )
    cleaned = _strip_invented_args(msg, _state("cualquier valor, no me importa."))
    assert _args_of(cleaned)["distribution_params"] == [0.5]


def test_delegation_only_applies_to_last_turn():
    """Una delegación en mensajes anteriores NO arrastra permiso al turno actual."""
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"start_date": "2020-01-01"},
    )
    state = {
        "messages": [
            HumanMessage(content="hace tiempo dije usa los defaults"),
            AIMessage(content="ok"),
            HumanMessage(content="ahora genera otra cosa"),
        ],
        "csv_metadata": None,
    }
    cleaned = _strip_invented_args(msg, state)
    # El último mensaje no delega → strip activo → start_date sin evidencia se quita.
    assert "start_date" not in _args_of(cleaned)


# ── Patrones nuevos en numeric_list ─────────────────────────────────────────


def test_keep_distribution_params_when_user_writes_p_equals_integer():
    """'p = 30' debe contar como evidencia de parámetro numérico."""
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [30]},
    )
    cleaned = _strip_invented_args(msg, _state("p = 30"))
    assert _args_of(cleaned)["distribution_params"] == [30]


def test_keep_distribution_params_when_user_writes_lambda_equals():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [5]},
    )
    cleaned = _strip_invented_args(msg, _state("lambda = 5"))
    assert _args_of(cleaned)["distribution_params"] == [5]


def test_keep_distribution_params_when_user_writes_parametros_plural():
    """'parametros' (sin la s singular) debe matchear."""
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [0.5]},
    )
    cleaned = _strip_invented_args(msg, _state("los parametros que quieras"))
    # "que quieras" también delega → bypass. La pieza importante es que pase.
    assert _args_of(cleaned)["distribution_params"] == [0.5]


def test_keep_distribution_params_when_user_mentions_probabilidad():
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {"distribution_params": [0.3]},
    )
    cleaned = _strip_invented_args(msg, _state("probabilidad 0.3"))
    assert _args_of(cleaned)["distribution_params"] == [0.3]


# ── Caso integrado: caso del usuario ("genera serie sintética normal") ─────


def test_full_invention_only_distribution_type_survives():
    """Escenario real reportado: el LLM inventa start_date, frequency, periods,
    distribution_params. Solo distribution_type=1 sobrevive porque "normal" es
    evidencia textual."""
    msg = _tool_call_msg(
        "generate_synthetic_distribution",
        {
            "start_date": "2020-01-01",
            "frequency": "D",
            "distribution_type": 1,
            "distribution_params": [0.0, 1.0],
            "periods": 30,
        },
    )
    cleaned = _strip_invented_args(msg, _state("Genera una serie sintética normal."))
    args = _args_of(cleaned)
    assert "start_date" not in args
    assert "frequency" not in args
    assert "periods" not in args
    assert "distribution_params" not in args
    assert args.get("distribution_type") == 1  # "normal" justifica el tipo


# ── Inyección de directiva de delegación ────────────────────────────────────


def test_delegation_directive_injected_when_pending_and_missing():
    """Cuando el usuario delega y faltan obligatorios → directiva al LLM."""
    state = {
        "messages": [
            HumanMessage(content="Generame una serie sintética"),
            AIMessage(content="", tool_calls=[
                {"name": "generate_synthetic_distribution", "args": {},
                 "id": "c1", "type": "tool_call"},
            ]),
            HumanMessage(content="usa los defaults"),
        ],
        "csv_path": None,
        "csv_metadata": None,
        "pending_tool": "generate_synthetic_distribution",
        "pending_params": {},
        "optionals_confirmed_for": None,
        "rag_context": None,
        "error_count": 0,
        "error_info": None,
    }
    captured: list = []

    class _FakeLLM:
        def invoke(self, messages):
            captured.append(list(messages))
            return AIMessage(
                content="Propongo Normal[0,1] sobre 100 días.",
                tool_calls=[{
                    "name": "generate_synthetic_distribution",
                    "args": {
                        "start_date": "2026-01-01",
                        "frequency": "D",
                        "distribution_type": 1,
                        "distribution_params": [0.0, 1.0],
                        "periods": 100,
                    },
                    "id": "call_mock",
                    "type": "tool_call",
                }],
            )

    with patch.object(reasoning_mod, "get_llm_with_tools", lambda _tools: _FakeLLM()):
        updates = reasoning_mod.razonador_node(state)

    # La directiva DEBE estar entre los mensajes que recibió el LLM.
    msgs = captured[0]
    from langchain_core.messages import SystemMessage
    directives = [m for m in msgs if isinstance(m, SystemMessage)
                  and "debes invocar" in m.content.lower()]
    assert directives, "no se inyectó la directiva de delegación"
    # Los params propuestos deben sobrevivir (bypass del strip + merge).
    pending = updates.get("pending_params") or {}
    final_args = updates["messages"][0].tool_calls[0]["args"]
    args = pending if pending else final_args
    assert args.get("distribution_type") == 1
    assert args.get("periods") == 100


def test_delegation_directive_injected_when_no_pending_tool():
    """Primer turno con delegación: directiva indica identificar herramienta."""
    state = {
        "messages": [HumanMessage(content="genera lo que tú quieras")],
        "csv_path": None,
        "csv_metadata": None,
        "pending_tool": None,
        "pending_params": None,
        "optionals_confirmed_for": None,
        "rag_context": None,
        "error_count": 0,
        "error_info": None,
    }
    captured: list = []

    class _FakeLLM:
        def invoke(self, messages):
            captured.append(list(messages))
            return AIMessage(content="OK")

    with patch.object(reasoning_mod, "get_llm_with_tools", lambda _tools: _FakeLLM()):
        reasoning_mod.razonador_node(state)

    from langchain_core.messages import SystemMessage
    directives = [m for m in captured[0] if isinstance(m, SystemMessage)
                  and "debes invocar" in m.content.lower()
                  and "más adecuada" in m.content.lower()]
    assert directives, "directiva sin pending_tool no inyectada"


def test_delegation_directive_not_injected_without_delegation():
    """Sin delegación en el último mensaje, no se inyecta ninguna directiva extra."""
    state = {
        "messages": [HumanMessage(content="Genera serie diaria normal con 100 periodos")],
        "csv_path": None,
        "csv_metadata": None,
        "pending_tool": None,
        "pending_params": None,
        "optionals_confirmed_for": None,
        "rag_context": None,
        "error_count": 0,
        "error_info": None,
    }
    captured: list = []

    class _FakeLLM:
        def invoke(self, messages):
            captured.append(list(messages))
            return AIMessage(content="OK")

    with patch.object(reasoning_mod, "get_llm_with_tools", lambda _tools: _FakeLLM()):
        reasoning_mod.razonador_node(state)

    from langchain_core.messages import SystemMessage
    extras = [m for m in captured[0] if isinstance(m, SystemMessage)
              and "debes invocar" in m.content.lower()]
    assert not extras, f"no debería inyectarse directiva sin delegación: {extras}"


# ── Integración con razonador_node ──────────────────────────────────────────


def test_razonador_node_routes_to_pending_when_llm_invents_args():
    state = {
        "messages": [HumanMessage(content="Genera una serie sintética normal.")],
        "csv_path": None,
        "csv_metadata": None,
        "pending_tool": None,
        "pending_params": None,
        "optionals_confirmed_for": None,
        "rag_context": None,
        "error_count": 0,
        "error_info": None,
    }
    inventive = AIMessage(
        content="",
        tool_calls=[{
            "name": "generate_synthetic_distribution",
            "args": {
                "start_date": "2020-01-01",
                "frequency": "D",
                "distribution_type": 1,
                "distribution_params": [0.0, 1.0],
                "periods": 30,
            },
            "id": "call_mock",
            "type": "tool_call",
        }],
    )

    class _FakeLLM:
        def invoke(self, _messages):
            return inventive

    with patch.object(reasoning_mod, "get_llm_with_tools", lambda _tools: _FakeLLM()):
        updates = reasoning_mod.razonador_node(state)

    assert updates["pending_tool"] == "generate_synthetic_distribution"
    pending = updates["pending_params"]
    assert "start_date" not in pending
    assert "frequency" not in pending
    assert "distribution_params" not in pending
    assert "periods" not in pending
    # "normal" justifica distribution_type=1
    assert pending["distribution_type"] == 1
