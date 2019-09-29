# docker run --runtime=nvidia --ipc=host --rm -v /data:/data -p 8899:8888 -d lab07


FROM nvcr.io/nvidia/pytorch:19.09-py3

RUN pip install albumentations

COPY berkeley-mids-w251-week07-lab.ipynb /workspace
EXPOSE 8888
CMD jupyter notebook  --no-browser --ip=0.0.0.0 --allow-root

