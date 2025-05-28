# syntax=docker/dockerfile:1.3

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy project metadata files for dependency resolution and source code
COPY src /app/src
COPY pyproject.toml /app/

# Install dependencies from pyproject.toml using uv
RUN uv pip install --system --no-cache-dir .

# Copy project metadata files for dependency resolution and source code
COPY . /app/

RUN --mount=type=secret,id=aws,target=/root/.aws/credentials uv run dvc pull

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI application
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "src/app/main.py"]