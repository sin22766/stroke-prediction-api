# Use distroless image with Python 3.12 and uv preinstalled
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy project metadata files for dependency resolution
COPY pyproject.toml /app/
COPY uv.lock /app/

# Install dependencies from pyproject.toml using uv
RUN uv pip install --system --no-cache-dir .

# Copy the full source code
COPY src /app/src

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]