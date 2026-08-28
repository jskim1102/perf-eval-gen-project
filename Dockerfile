# syntax=docker/dockerfile:1.7
# perf-eval-gen — FastAPI 백엔드 + stdlib 정적 프론트, 둘 다 같은 이미지에서 뜬다.
# 베이스: python:3.12-slim (호스트 .venv 와 같은 3.12 계열 — numpy 2.5 / pandas 3.0 요구).
# torch 는 requirements.txt 의 cu128 휠 → 이미지에 CUDA 런타임이 들어간다(약 9GB).
# GPU 는 compose 의 nvidia device reservation 으로 붙인다(호스트에 nvidia-container-toolkit 필요).

FROM python:3.12-slim

# libgomp1 = torch/scikit-image OpenMP 런타임. 그 외 시스템 의존 없음(opencv 미사용).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HOME/USER: 컨테이너는 호스트 uid 로 도는데 그 uid 의 passwd 항목이 없다.
# torch(_dynamo → getpass.getuser())가 pwd 조회로 죽으므로 USER 를 env 로 준다
# (python 3.12 getpass 는 LOGNAME/USER 환경변수를 먼저 본다). HOME 은 캐시 경로용.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    USER=app

WORKDIR /app

# 의존성 먼저 — 코드만 바뀔 때 이 레이어(수 GB)는 재사용된다.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# 애플리케이션 소스. dataset/huggingface/runs 는 이미지에 넣지 않는다(.dockerignore).
COPY web ./web
COPY scripts ./scripts
COPY metrics ./metrics
COPY tests ./tests

# 호스트 umask 077 로 만들어진 소스는 0600 으로 복사된다 — compose 는 비-root
# (호스트 uid) 로 돌기 때문에 읽기 권한을 열어줘야 import 가 된다.
RUN chmod -R a+rX /app

EXPOSE 8000
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
