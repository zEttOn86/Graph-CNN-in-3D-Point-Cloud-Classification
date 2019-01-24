# Auther tozawa
# RHUMBA
FROM nvidia/cuda:9.0-cudnn7-devel

# RUN 命令はイメージの中で実行するコマンドを指示します
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    sudo git wget curl unzip tree graphviz bzip2 eog && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# userの追加
RUN groupadd -g 1001 developer && \
    useradd -g developer -u 1001 -G sudo -m -s /bin/bash penguin && \
    echo "penguin:piyopiyo" | chpasswd && \
    echo "penguin ALL=(ALL) ALL" >> /etc/sudoers

USER penguin
WORKDIR /home/penguin/

# https://towardsdatascience.com/docker-for-data-science-9c0ce73e8263
RUN wget https://repo.continuum.io/archive/Anaconda2-5.3.1-Linux-x86_64.sh && \
    bash Anaconda2-5.3.1-Linux-x86_64.sh -b && \
    rm Anaconda2-5.3.1-Linux-x86_64.sh

# Set PATH to conda
ENV PATH /home/penguin/anaconda2/bin:$PATH

# Update pip and anaconda packages
RUN conda update conda && \
    conda update --all && \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow-gpu==1.12.0 \
    SimpleITK==1.1.0 \
    tqdm==4.19.4 \
    gensim==2.1.0
