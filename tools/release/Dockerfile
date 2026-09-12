ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_aarch64:latest
FROM ${MANYLINUX_IMAGE}
ARG TARGETARCH
ARG RUST_VERSION=1.94.1
ARG NODE_VERSION=22.21.1
ENV PATH=/root/.cargo/bin:/opt/python/cp312-cp312/bin:/opt/node/bin:$PATH
RUN dnf install -y perl-core pkgconfig openssl-devel clang git curl xz procps-ng \
    && dnf clean all
RUN curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && rustup component add rustfmt clippy
RUN case "$TARGETARCH" in arm64) arch=arm64 ;; amd64) arch=x64 ;; *) exit 1 ;; esac \
    && cd /tmp \
    && curl --proto '=https' --tlsv1.2 -fsSLO https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${arch}.tar.xz \
    && curl --proto '=https' --tlsv1.2 -fsSLO https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt \
    && awk "/ node-v${NODE_VERSION}-linux-${arch}.tar.xz\$/" SHASUMS256.txt | sha256sum --check --strict \
    && mkdir /opt/node \
    && tar -xJf node-v${NODE_VERSION}-linux-${arch}.tar.xz --strip-components=1 -C /opt/node
RUN dnf install -y ripgrep && dnf clean all
WORKDIR /work
ENTRYPOINT ["python3", "tools/release/release.py", "build-target"]
