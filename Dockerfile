FROM nexus.maple.maceroc.com/docker_millegrilles/millegrilles/messages_python:2026.7.30 AS stage1

ARG APT_PROXY
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

ENV PIP_INDEX_URL="$PIP_INDEX_URL" \
    PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST"

# Install dependencies
RUN if [ -n "$APT_PROXY" ]; then \
         echo "Acquire::http::Proxy \"$APT_PROXY\";" > /etc/apt/apt.conf.d/01proxy; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Stage 2
FROM stage1 AS stage2

ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

# Creer repertoire app, copier fichiers
COPY ./requirements.txt $BUILD_FOLDER/requirements.txt

RUN pip3 install --no-cache-dir -r $BUILD_FOLDER/requirements.txt

# Pour offline build
#ENV PIP_FIND_LINKS=$BUILD_FOLDER/pip \
#    PIP_RETRIES=0 \
#    PIP_NO_INDEX=true

# Final stage
FROM stage2

ARG VBUILD=2026.7.0
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

ENV KEY_PATH=/run/secrets/key_cert.pem \
    CA_PATH=/run/secrets/pki.millegrille.cert \
    MQ_HOSTNAME=mq \
    MQ_PORT=5673 \
    REDIS_HOSTNAME=redis \
    REDIS_PASSWORD_PATH=/var/run/secrets/passwd.redis.txt \
    WEB_PORT=1443 \
    OLLAMA_URL=http://ollama:11434

EXPOSE 80 443

COPY . $BUILD_FOLDER

RUN cd $BUILD_FOLDER/  && \
    python3 ./setup.py install

CMD ["-m", "millegrilles_ollama_relai"]
