FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessing.py .
COPY train_resnet18_tinyimagenet.py .

RUN mkdir -p /app/outputs

CMD ["python", "train_resnet18_tinyimagenet.py"]
