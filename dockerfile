FROM python:3.12-slim

WORKDIR /app

COPY ./ .

RUN pip install .

EXPOSE 8100

CMD ["fastapi","run","src/app/main.py"]