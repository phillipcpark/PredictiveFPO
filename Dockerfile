FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    vim -y \
    git -y \
    python3-pip -y \
    unzip -y

RUN pip3 install 'dgl==0.5.3' \
                 'dgl-cu110==0.5.3' \
                 mpmath \
                 torch

RUN git clone https://github.com/phillipcpark/PredictiveFPO.git

RUN cd PredictiveFPO/resources/public/ds && \
    unzip tstredux128_mpuntie8_resblock2_hidden32_ep174.zip && \
    cd /PredictiveFPO



