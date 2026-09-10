FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vhs_refactor ./vhs_refactor
COPY number_card_60x60 ./number_card_60x60

# HF Spaces runs containers as a non-root user with no fixed UID; make everything writable.
RUN chmod -R a+rwX /app

EXPOSE 7860

CMD ["voila", "vhs_refactor/report.ipynb", "--port=7860", "--no-browser", "--Voila.ip=0.0.0.0"]
