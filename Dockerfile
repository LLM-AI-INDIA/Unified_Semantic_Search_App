# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10


EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_GLOBAL_DEVELOPMENTMODE=false

# Install production dependencies.
RUN pip install -r requirements.txt

# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
#CMD streamlit run --server.port 8080 --server.enableCORS false app.py

# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD streamlit run --server.port $STREAMLIT_SERVER_PORT --server.enableCORS $STREAMLIT_SERVER_ENABLE_CORS --global.developmentMode $STREAMLIT_GLOBAL_DEVELOPMENTMODE app.py