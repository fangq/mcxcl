FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y wget devtoolset-7-gcc-c++ devtoolset-7-gcc && yum remove -y devtoolset-10-gcc-c++ devtoolset-10-gcc \
    && yum install -y ocl-icd ocl-icd-devel && yum clean all
ENV PATH /opt/rh/devtoolset-7/root/usr/bin:${PATH}
COPY build_wheels.sh /
WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]
