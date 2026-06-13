FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY enums.py ./
COPY common.py ./
COPY environment_types.py ./
COPY agent_types.py ./
COPY simulation_types.py ./
COPY sim_engine.py ./
COPY dashboard_cli.py ./
COPY substrate_architecture.py ./

CMD ["python", "sim_engine.py"]
