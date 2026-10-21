# StS2 RL training image — kaiwu-style actor/learner architecture.
#
#   stage 1 (native):  .NET 9 SDK + clang -> NativeAOT linux-x64 engine (.so)
#   stage 2 (runtime): python 3.13 slim + torch CPU + sts2_gym + kaiwu_sts2
#
# Build from the sts2-rl workspace root:
#   docker build -f docker/Dockerfile -t sts2-rl:latest .
#
# If download.pytorch.org is slow in your network, pre-pull the torch wheel or
# override the index, e.g.:
#   docker build --build-arg TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cpu ...

FROM mcr.microsoft.com/dotnet/sdk:9.0-bookworm-slim AS native
# Switch apt to a China-friendly mirror first (best-effort; no-op elsewhere).
RUN set -e; \
    (sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/debian.sources 2>/dev/null || true); \
    apt-get update \
 && apt-get install -y --no-install-recommends clang zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY emulator/src/Sts2Emulator ./Sts2Emulator
COPY scripts/relink-linux.sh ./relink-linux.sh
# Publish compiles the AOT object file; the ILC-generated ELF version script
# hides every export (ILC 9.x limitation), so finish with a manual re-link
# that exports the Sts2_* API via our own version script + defsym aliases.
RUN dotnet publish ./Sts2Emulator/Sts2Emulator.csproj \
      -c Release -r linux-x64 --self-contained -o /out \
 && bash ./relink-linux.sh /build/Sts2Emulator /out

FROM python:3.13-slim AS runtime
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STS2_LIB_PATH=/app/native

RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch \
 && pip install --no-cache-dir --index-url ${PIP_INDEX} gymnasium numpy pyzmq msgpack

COPY --from=native /out/Sts2Emulator.so /app/native/Sts2Emulator.so
COPY emulator/src/sts2_gym /app/sts2_gym
COPY kaiwu_sts2 /app/kaiwu_sts2
COPY conf /app/conf

# ROLE selects the process (modelpool | env_server | actor | learner | aisrv | smoke)
ENTRYPOINT ["python", "/app/kaiwu_sts2/start.py"]
