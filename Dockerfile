FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-filter-repo \
    curl \
    wget \
    openssh-client \
    vim \
    sudo \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace

USER $USERNAME
WORKDIR /workspace