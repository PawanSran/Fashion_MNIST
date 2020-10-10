FROM python:3.6

# set the working directory
RUN ["mkdir", "fashion_mnist_app"]
WORKDIR /fashion_mnist_app

ADD . /fashion_mnist_app

# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]


EXPOSE 5000

ENTRYPOINT ["python" , "app.py"]