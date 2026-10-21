ARG BASE_IMAGE=quay.io/pypa/manylinux2014_x86_64
FROM ${BASE_IMAGE}
# the CentOS 7 aarch64 repositories no longer carry ocl-icd; when it is missing, build the
# Khronos OpenCL headers and ICD loader, which provide the same libOpenCL.so.1
RUN yum install -y vim-common && \
    if ! yum install -y ocl-icd ocl-icd-devel; then \
        git clone --depth 1 --branch v2023.12.14 https://github.com/KhronosGroup/OpenCL-Headers.git /tmp/clheaders && \
        cmake -S /tmp/clheaders -B /tmp/clheaders/build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
        cmake --install /tmp/clheaders/build && \
        git clone --depth 1 --branch v2023.12.14 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git /tmp/clloader && \
        cmake -S /tmp/clloader -B /tmp/clloader/build -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_PREFIX_PATH=/usr -DBUILD_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
        cmake --build /tmp/clloader/build -j4 && cmake --install /tmp/clloader/build && \
        rm -rf /tmp/clheaders /tmp/clloader; \
    fi && yum clean all
ENV PATH /opt/rh/devtoolset-7/root/usr/bin:${PATH}
COPY build_wheels.sh /
WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]
