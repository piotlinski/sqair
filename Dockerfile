FROM nvcr.io/nvidia/pytorch:22.09-py3

COPY . /workspace
RUN pip install -r requirements.txt
