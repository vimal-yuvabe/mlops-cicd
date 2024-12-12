FROM python:3.9

WORKDIR /crjob

COPY . .

RUN pip install --no-cache-dir --upgrade -r /crjob/requirements.txt

CMD [ "python","pipeline.py" ]