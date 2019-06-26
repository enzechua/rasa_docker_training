FROM python:3.6-slim

WORKDIR /app
ADD . .
RUN pip install rasa_core_sdk
EXPOSE 5055

CMD python -m rasa_core_sdk.endpoint --actions actions
