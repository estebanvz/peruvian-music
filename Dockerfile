FROM tensorflow/tensorflow:2.9.0rc2-gpu-jupyter
RUN add-apt-repository ppa:mscore-ubuntu/mscore-stable
RUN apt update
RUN apt-get install musescore -y
RUN pip install music21
RUN pip install scikit-learn
RUN apt install timidity
RUN pip install pandas