FROM python:3.6

# set the working directory
RUN ["mkdir", "fashion_mnist_app"]
WORKDIR /fashion_mnist_app

ADD . /fashion_mnist_app

#fixing libgl error
#RUN apt-get update ##[edited]
#RUN apt-get install 'ffmpeg'\
#    'libsm6'\ 
#    'libxext6'  -y


# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

COPY "fashion_mnist.ipynb" .

EXPOSE 8080
CMD ["python", "app.py"]

# install environment dependencies
#COPY "catalog-screencap.png" .

# provision environment

#ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8080", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]