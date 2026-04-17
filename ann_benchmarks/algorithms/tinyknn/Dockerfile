FROM ann-benchmarks

RUN git clone https://github.com/thomasahle/tinyknn.git
RUN python3 -m pip install 'Cython<3' 'numpy<2'
RUN cd tinyknn && PIP_NO_BUILD_ISOLATION=1 python3 -m pip install --no-build-isolation .
RUN python3 -c 'import tinyknn'
