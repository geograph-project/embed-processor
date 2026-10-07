FROM python:3.11-slim
LABEL maintainer="Manticore Software Ltd. <contact@manticoresearch.com>"

COPY requirements.txt .
RUN apt-get update && apt-get install -y git pkg-config libsentencepiece-dev && \
	pip install --no-cache-dir -r requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple/ \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY entrypoint /entrypoint

# Set execute permissions for the entrypoint script
RUN chmod +x /entrypoint

ENTRYPOINT ["/entrypoint"]
WORKDIR /src

