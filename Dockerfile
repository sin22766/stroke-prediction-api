FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy project metadata files for dependency resolution and source code
COPY pyproject.toml /app/
COPY uv.lock /app/
COPY src /app/src

# Install dependencies from pyproject.toml using uv
RUN uv pip install --system --no-cache-dir .

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI application
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "src/app/main.py"]