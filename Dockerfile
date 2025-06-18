FROM registry.millegrilles.com/millegrilles/messages_python:2025.4.106 as stage1

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Stage 2
FROM stage1 as stage2

# Creer repertoire app, copier fichiers
COPY ./requirements.txt $BUILD_FOLDER/requirements.txt

RUN pip3 install --no-cache-dir -r $BUILD_FOLDER/requirements.txt

# Pour offline build
#ENV PIP_FIND_LINKS=$BUILD_FOLDER/pip \
#    PIP_RETRIES=0 \
#    PIP_NO_INDEX=true

# Final stage
FROM stage2

ARG VBUILD=2025.3.0

ENV CERT_PATH=/run/secrets/cert.pem \
    KEY_PATH=/run/secrets/key.pem \
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
