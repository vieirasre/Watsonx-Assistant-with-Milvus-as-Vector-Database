FROM python:3.11.2-slim-bullseye


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

RUN \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    python3 -c 'from datasets import load_dataset;load_dataset("ruslanmv/ai-medical-chatbot");from towhee import ops;ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base")'

EXPOSE 8080

CMD ["python", "app.py"]
