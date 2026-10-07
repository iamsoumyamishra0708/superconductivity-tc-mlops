FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Poora project copy kar rahe hain taaki api, frontend aur models sabhi access ho sakein
COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]