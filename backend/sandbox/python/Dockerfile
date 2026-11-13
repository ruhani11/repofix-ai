# Trusted runtime build context only: backend/sandbox/python.
# The operator builds explicitly; validation never builds repository Dockerfiles.
FROM python:3.12-slim

LABEL ai.repofix.validation-profile="python-pytest-v1"
LABEL ai.repofix.validation-runtime="3"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY sandbox-runtime-requirements.txt /opt/repofix/sandbox-runtime-requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    -r /opt/repofix/sandbox-runtime-requirements.txt \
    && python -m pip check \
    && mkdir -p /workspace \
    && chown 10001:10001 /workspace

COPY sandbox_runtime.py /opt/repofix/sandbox_runtime.py
COPY pytest.ini /opt/repofix/pytest.ini
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python", "-I", "-B", "/opt/repofix/sandbox_runtime.py"]
CMD ["--self-check"]
