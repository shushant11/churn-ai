export PYTHONPATH := src
install:      ; pip install -r requirements.txt
install-all:  ; pip install -r requirements.txt -r requirements-extra.txt
train:        ; python -m churnlab.train
train-fast:   ; python -m churnlab.train --fast
test:         ; pytest -q
eval:         ; python -m churnlab.eval_llm
eval-llm:     ; python -m churnlab.eval_llm --llm --judge
api:          ; uvicorn churnlab.api:app --reload
app:          ; streamlit run app/streamlit_app.py
mcp:          ; python -m churnlab.mcp_server
mlflow:       ; mlflow ui
docker:       ; docker compose up --build
