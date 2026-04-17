FROM ann-benchmarks

RUN apt-get update && apt-get install -y cmake libboost-all-dev libeigen3-dev libgsl0-dev
RUN pip3 install pybind11 nmslib
RUN python3 -c 'import nmslib'
