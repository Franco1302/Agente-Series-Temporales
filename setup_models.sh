#!/usr/bin/env bash
# Descarga los modelos óptimos para tool calling con 6 GB VRAM (Q4_K_M).
# Uso: bash setup_models.sh [--solo <modelo>]
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── Modelos seleccionados ────────────────────────────────────────────────────
# Etiqueta exacta en Ollama Hub · VRAM estimada con Q4_K_M
declare -A MODELS_DESC=(
    ["qwen2.5:7b-instruct-q4_K_M"]="Qwen2.5 7B Instruct  · ~4.5 GB VRAM · Mejor equilibrio tool-calling/razonamiento"
    ["llama3.1:8b-instruct-q4_K_M"]="Llama 3.1 8B Instruct · ~4.9 GB VRAM · Function-calling nativo de Meta"
    ["qwen2.5-coder:7b-instruct-q4_K_M"]="Qwen2.5-Coder 7B     · ~4.5 GB VRAM · Excelente generación de JSON estructurado"
    ["mistral:7b-instruct-q4_K_M"]="Mistral 7B Instruct  · ~4.1 GB VRAM · Sólido en tool-calling y seguimiento de instrucciones"
)

MODELS=(
    "qwen2.5:7b-instruct-q4_K_M"
    "llama3.1:8b-instruct-q4_K_M"
    "qwen2.5-coder:7b-instruct-q4_K_M"
    "mistral:7b-instruct-q4_K_M"
)

# ── Verificaciones previas ───────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo -e "${RED}ERROR: ollama no está instalado o no está en PATH.${NC}"
    exit 1
fi

if ! ollama list &>/dev/null; then
    echo -e "${RED}ERROR: El servicio Ollama no está corriendo. Ejecuta 'ollama serve' primero.${NC}"
    exit 1
fi

# ── Modo --solo ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--solo" && -n "${2:-}" ]]; then
    MODELS=("$2")
fi

# ── Descarga ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}=== Setup de modelos para MCP tool calling ===${NC}"
echo ""

TOTAL=${#MODELS[@]}
COUNT=0
FAILED=()

for model in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    desc="${MODELS_DESC[$model]:-Sin descripción}"
    echo -e "${YELLOW}[$COUNT/$TOTAL] Descargando: ${NC}${desc}"
    echo "      Tag: $model"
    echo ""

    if ollama pull "$model"; then
        echo -e "${GREEN}  ✓ $model descargado correctamente${NC}"
    else
        echo -e "${RED}  ✗ Error descargando $model${NC}"
        FAILED+=("$model")
    fi
    echo ""
done

# ── Resumen ──────────────────────────────────────────────────────────────────
echo -e "${YELLOW}=== Resumen ===${NC}"
echo "Modelos disponibles en Ollama:"
ollama list | grep -E "$(IFS='|'; echo "${MODELS[*]}")" || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}Modelos con error:${NC}"
    for m in "${FAILED[@]}"; do echo "  - $m"; done
    exit 1
fi

echo -e "\n${GREEN}Todos los modelos descargados. Ejecuta ahora:${NC}"
echo "  python mcp_benchmark.py"
