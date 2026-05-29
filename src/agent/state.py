"""Definición del estado compartido del agente LangGraph."""

from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Estado completo que fluye a través de todos los nodos del grafo.

    LangGraph pasa este dict entre nodos; cada nodo puede leer y devolver
    una actualización parcial. El campo `messages` usa `add_messages` como
    reducer, lo que garantiza que los mensajes se acumulen en lugar de
    sobreescribirse.
    """

    # Historial completo de mensajes (HumanMessage, AIMessage, ToolMessage…)
    messages: Annotated[list[BaseMessage], add_messages]

    # Ruta al CSV cargado por el usuario desde el sidebar (None si no hay fichero)
    csv_path: Optional[str]

    # Columnas, nº de filas y dtypes inferidos del CSV activo
    csv_metadata: Optional[dict]

    # Nombre de la herramienta en espera de completar sus parámetros
    pending_tool: Optional[str]

    # Parámetros recogidos hasta el momento para la herramienta pendiente
    # {nombre_param: valor_o_None}; None cuando no hay herramienta pendiente
    pending_params: Optional[dict]

    # Nombre de la herramienta cuyos parámetros opcionales (tunables) ya se
    # confirmaron con el usuario. Evita el bucle "preguntar → respuesta del
    # usuario → preguntar de nuevo" cuando se ejecuta el siguiente ciclo del
    # razonador.
    optionals_confirmed_for: Optional[str]

    # Fragmentos de documentación recuperados por el nodo recuperar_contexto
    rag_context: Optional[str]

    # Contador de errores consecutivos; limita el bucle de reintentos (máx. 3)
    error_count: int

    # Descripción estructurada del último error para informar al usuario
    error_info: Optional[str]

    # Memoria estructurada de la sesión usada para la herencia genérica de
    # parámetros entre herramientas. Esquema:
    #   "by_param": dict[param_name -> {value, source_tool, turn}]
    #       Vista pivotada por parámetro: cada entrada apunta al último valor
    #       observado en una ejecución exitosa, con qué tool lo produjo y en
    #       qué turno (índice ordinal). Solo se registran parámetros que
    #       pertenecen a alguna familia semántica heredable
    #       (`src.agent.param_families.INHERITABLE_PARAMS`), nunca params de
    #       dominio que cambien la semántica del análisis.
    #   "by_tool":  dict[tool_name -> {args_completos_de_la_ultima_ejecucion}]
    #       Snapshot completo de la última ejecución exitosa de cada tool.
    #       Útil para auditoría y para escenarios "repítelo" dirigidos a una
    #       tool concreta.
    session_facts: Optional[dict]
