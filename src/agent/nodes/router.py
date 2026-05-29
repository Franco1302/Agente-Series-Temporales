"""Nodo router: clasifica la intención del usuario antes del razonador.

El modelo cuantizado pequeño enruta mal en modo "auto": ante una petición de
acción tiende a responder en prosa en vez de emitir la tool call, y a menudo
ignora `tool_choice="any"`. Este nodo sustituye el antiguo gate por palabras
clave (frágil: "amplía" no matcheaba "aument") por una clasificación de la
intención mediante una única llamada LLM barata, que además **elige la
herramienta** asociada. El razonador usa esa elección como última red: si el
modelo principal no emite la tool call ni forzado, construye la llamada de esa
herramienta de forma determinista y deja que `solicitar_parametros_node` recoja
los datos. Así una petición de acción nunca termina en un callejón de prosa.

La clasificación razona sobre el PROPÓSITO de la petición. El LLM responde una
de estas palabras:

  * ``generar``  → crear una serie/datos nuevos desde cero (sintéticos).
  * ``drift``    → comprobar si los datos han cambiado / detectar drift.
  * ``aumentar`` → ampliar una serie existente con más observaciones.
  * ``predecir`` → pronosticar valores futuros (forecast, SARIMAX).
  * ``variable`` → crear una columna/variable nueva derivada de la serie.
  * ``teoria``   → pregunta conceptual / explicación.
  * ``texto``    → saludo, capacidades, o nada de lo anterior.

Las cinco primeras mapean a una herramienta analítica concreta (intent
``analisis``); ``teoria`` mapea a ``consultar_teoria``; ``texto`` no fuerza nada.

Cortocircuito determinista: si hay una recogida de parámetros en curso
(``pending_tool``), no se gasta una llamada en clasificar: se devuelve
``continuacion`` y el razonador aplica su lógica de replay/merge existente.
"""

from __future__ import annotations

import time
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.config.llm_config import get_chat_ollama
from src.observability import emit_llm_call

#: Intenciones de alto nivel que el razonador sabe interpretar.
VALID_INTENTS: tuple[str, ...] = ("analisis", "teoria", "texto")

#: Raíz de cada categoría del router → herramienta analítica asociada. El orden
#: importa: se comprueba por orden y gana la primera raíz contenida en la salida
#: del LLM. La raíz es un prefijo tolerante a la conjugación ("gener" cubre
#: "generar"/"genera"; "predec"/"predic" cubre "predecir"/"predicción").
_CATEGORY_TOOL: tuple[tuple[str, str], ...] = (
    ("gener", "generate_synthetic_distribution"),
    ("drift", "detect_drift"),
    ("deriva", "detect_drift"),
    ("aument", "augment_time_series"),
    ("amplia", "augment_time_series"),
    ("predec", "forecast_time_series"),
    ("predic", "forecast_time_series"),
    ("pronos", "forecast_time_series"),
    ("forecast", "forecast_time_series"),
    ("variable", "create_exogenous_variable"),
    ("exogen", "create_exogenous_variable"),
)

_ROUTER_SYSTEM = """\
Clasifica la ÚLTIMA petición del usuario en UNA sola palabra, sin explicaciones.

- generar: crear una serie o datos nuevos desde cero (sintéticos: distribución,
  tendencia, estacional, ARMA), sin partir de un fichero.
- drift: comprobar si la distribución de los datos ha cambiado / detectar drift.
- aumentar: ampliar o aumentar una serie existente con más observaciones.
- predecir: pronosticar o predecir valores futuros (forecast, SARIMAX).
- variable: crear una variable o columna nueva derivada de la serie.
- teoria: pregunta por un concepto o pide una explicación (qué es, qué
  significa, diferencias entre métodos, cómo funciona algo).
- texto: saludo, conversación, pregunta sobre tus capacidades, o nada de lo
  anterior.

Responde solo con una de estas palabras: generar, drift, aumentar, predecir,
variable, teoria, texto."""


def _last_human_text(messages: list) -> str:
    """Devuelve el contenido del último HumanMessage del historial, o ''."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _strip_accents(text: str) -> str:
    """Normaliza quitando tildes para que el parser tolere 'predecir'/'teoría'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def parse_router_output(text: str) -> tuple[str, str | None]:
    """Mapea la salida del LLM a ``(intent, intent_tool)`` (degradación segura).

    ``intent`` ∈ ``{"analisis", "teoria", "texto"}``. ``intent_tool`` es el nombre
    de la herramienta asociada (None para ``texto``). Tolera ruido y tildes
    buscando la raíz de cada categoría. Si nada encaja, devuelve ``texto`` (no se
    fuerza nada, igual que el comportamiento previo al router).
    """
    t = _strip_accents((text or "").strip().lower())
    for root, tool in _CATEGORY_TOOL:
        if root in t:
            return "analisis", tool
    if "teor" in t:
        return "teoria", "consultar_teoria"
    # Aceptamos también la palabra genérica "analisis" por robustez: si el modelo
    # responde la categoría de alto nivel en vez de una específica, asumimos la
    # generación sintética (la acción más común sin fichero de entrada).
    if "anal" in t:
        return "analisis", "generate_synthetic_distribution"
    return "texto", None


def clasificar_intencion_node(state: AgentState) -> dict:
    """Clasifica la intención del turno → ``state['intent']`` y ``['intent_tool']``."""
    # Recogida de parámetros en curso: el razonador ya sabe qué hacer.
    if state.get("pending_tool"):
        return {"intent": "continuacion", "intent_tool": None}

    last_human = _last_human_text(state.get("messages", []))
    if not last_human:
        return {"intent": "texto", "intent_tool": None}

    convo = [
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=last_human),
    ]
    try:
        # Temperatura 0: clasificación determinista, independiente de la
        # temperatura de síntesis configurada para las respuestas en prosa.
        llm = get_chat_ollama(temperature=0.0)
        t0 = time.perf_counter()
        raw = llm.invoke(convo)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        intent, intent_tool = parse_router_output(content)
        emit_llm_call(
            name="router.clasificar",
            messages=convo,
            response_raw=raw,
            response_final=raw,
            duration_ms=duration_ms,
        )
    except Exception:
        # Si el router falla, degradamos a "texto": el razonador no fuerza y el
        # modelo decide en modo auto (comportamiento previo al router).
        intent, intent_tool = "texto", None

    return {"intent": intent, "intent_tool": intent_tool}
