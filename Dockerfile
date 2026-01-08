FROM python:3.12-slim

WORKDIR /app

COPY docker_requirements.txt .

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r /app/docker_requirements.txt

COPY checkpoints/best_model.pt ./checkpoints/best_model.pt
COPY src ./src

COPY datasets/MovieLens_Large/dataset_stats.json ./datasets/MovieLens_Large/dataset_stats.json

COPY recsys_project ./recsys_project
COPY pyproject.toml ./pyproject.toml
RUN pip install -e .

ENTRYPOINT ["python", "-m", "src.predict"]