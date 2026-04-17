FROM ann-benchmarks

RUN apt-get update
RUN apt-get install -y wget gnupg git cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-dev python3 python3-setuptools python3-pip
RUN pip3 install pybind11 numpy

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
RUN sh -c 'echo deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main > /etc/apt/sources.list.d/oneapi.list'
RUN apt-get update
RUN apt-get install -y intel-oneapi-mkl intel-oneapi-mkl-devel

ENV MKLROOT=/opt/intel/oneapi/mkl/latest

RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      ${MKLROOT}/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    ${MKLROOT}/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    ${MKLROOT}/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  ${MKLROOT}/lib/intel64/libmkl_rt.so 150

RUN echo "/opt/intel/oneapi/compiler/latest/lib"        >  /etc/ld.so.conf.d/mkl.conf
RUN echo "/opt/intel/oneapi/compiler/latest/lib/intel64_lin" >> /etc/ld.so.conf.d/mkl.conf
RUN echo "${MKLROOT}/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf
RUN ldconfig
RUN echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

RUN git clone --single-branch --branch python_bindings https://github.com/microsoft/diskann
RUN mkdir -p diskann/build
RUN bash -lc 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && cd diskann/build && cmake -DCMAKE_BUILD_TYPE=Release ..'
RUN bash -lc 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && cd diskann/build && make -j'
RUN cd diskann/python && pip install -e .
RUN python3 -c 'import vamanapy'
