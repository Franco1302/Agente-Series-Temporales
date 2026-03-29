# Registro de errores y soluciones

## Como registrar errores nuevos

Usa este formato para cada error detectado en desarrollo:

```text
Fecha: YYYY-MM-DD
Error: mensaje principal
Contexto: donde ocurrio
Causa raiz: por que paso
Solucion aplicada: que se hizo
Estado: resuelto | pendiente
```

## Errores encontrados

### 2026-03-29 - No module named src

- Contexto: ejecucion de Streamlit desde la carpeta ui o sin PYTHONPATH.
- Causa raiz: Python no encuentra el paquete src fuera de la raiz del proyecto.
- Solucion aplicada:

```bash
cd /home/franco/Documentos/TFG
source .venv/bin/activate
PYTHONPATH=. streamlit run ui/app.py
```

- Estado: resuelto.

### 2026-03-29 - File does not exist: app.py

- Contexto: comando streamlit run app.py desde la raiz del repo.
- Causa raiz: el archivo de entrada esta en ui/app.py, no en la raiz.
- Solucion aplicada:

```bash
streamlit run ui/app.py
```

- Estado: resuelto.

### 2026-03-29 - File does not exist: streamlit_app.py

- Contexto: ejecucion de streamlit run sin target.
- Causa raiz: Streamlit busca streamlit_app.py por defecto si no se pasa archivo.
- Solucion aplicada:

```bash
PYTHONPATH=. streamlit run ui/app.py
```

- Estado: resuelto.

### 2026-03-29 - pip install requirements.txt

- Contexto: instalacion de dependencias.
- Causa raiz: faltaba la bandera -r, pip intento instalar un paquete llamado requirements.txt.
- Solucion aplicada:

```bash
pip install -r requirements.txt
```

- Estado: resuelto.

### 2026-03-29 - Error de conexion con el LLM local

- Contexto: Streamlit mostraba fallo en invocacion al LLM.
- Causa raiz: combinacion de sesion previa sin reinicio y nombre de modelo no alineado con Ollama.
- Solucion aplicada:
  - Ajustar OLLAMA_MODEL en .env a llama3.1:latest.
  - Reiniciar Streamlit despues de cambiar .env.
  - Validar servicio con curl a /api/tags y prueba de generacion.
- Estado: resuelto.