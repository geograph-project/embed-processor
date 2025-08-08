FROM python:3.9-slim
LABEL maintainer="Manticore Software Ltd. <contact@manticoresearch.com>"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple/

COPY entrypoint /entrypoint

# Set execute permissions for the entrypoint script
RUN chmod +x /entrypoint

ENTRYPOINT ["/entrypoint"]
WORKDIR /src

