FROM rust:bookworm AS builder
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y libclang-dev && rm -rf /var/lib/apt/lists/*
ENV LIBCLANG_PATH "/usr/lib/x86_64-linux-gnu/libclang-14.so"
ENV SQLX_OFFLINE=true
RUN cargo install --path faces

from debian:bookworm as run
COPY --from=builder /usr/local/cargo/bin/faces /usr/local/bin/faces
COPY --from=builder /app/models/* /usr/local/bin/models/
ENTRYPOINT ["/usr/local/bin/faces", "-v", "-e", "/usr/local/bin/models/arcface-int8.onnx", "-d", "/usr/local/bin/models/ultraface-int8.onnx"]
