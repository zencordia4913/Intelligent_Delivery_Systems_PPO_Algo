FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["gunicorn", "your_project_name.wsgi:application", "--bind", "0.0.0.0:8001"]
