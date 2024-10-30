FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN python3 -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libb64-0d
RUN apt-get install -y ffmpeg
RUN pip install torch 
RUN pip install transformers 
RUN pip install numpy 
RUN pip install tritonclient[all]
RUN pip install matplotlib
RUN pip install tiktoken
RUN pip install transformers_stream_generator
RUN pip install einops
RUN pip install accelerate>=0.26.0
RUN git clone https://github.com/triton-inference-server/python_backend -b r23.10
RUN mkdir data
RUN cd python_backend

COPY ./models /models
COPY ./data /data

# CMD ["tritonserver", "--model-repository=/models", "--log-format=default", "--log-file=/opt/tritonserver/logfile.log"] 
# CMD ["tritonserver", "--model-repository=/models", "--cache-config=local,size=1000048576", "--log-verbose=1"]   
# CMD ["tritonserver", "--model-repository=/models", "--log-format=default", "--log-file=/opt/tritonserver/logfile.log"]   
# CMD ["tritonserver", "--model-repository=/models", "--log-verbose=2"]
CMD ["tritonserver", "--model-repository=/models"]   
