FROM huggingface/trl-latest-gpu:latest

RUN mkdir /workspace
WORKDIR /workspace

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /workspace

USER user
# user can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace/

RUN source activate trl && \
    python3 -m pip install -U --no-build-isolation --no-cache-dir -r requirements.txt

RUN echo 'export PATH="$PATH:/home/user/.local/bin"' >> ~/.bashrc && \
    echo 'source activate trl' >> ~/.bashrc

# COPY jupyter_server_config.json /home/user/.jupyter/jupyter_server_config.json

WORKDIR /workspace

CMD ["/bin/bash"]
