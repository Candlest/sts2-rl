# StS2 RL 容器化架构（参考腾讯开悟 kaiwudrl）

在 `../README.md` 的三层架构之上，训练层被拆成开悟风格的 actor/learner 容器集群。
设计对照表（kaiwudrl → kaiwu_sts2）：

| kaiwudrl 概念 | kaiwu_sts2 实现 | 说明 |
|---|---|---|
| `aisrv`（对局编排，跑 rollout 工作流） | `kaiwu_sts2.aisrv` | 每 pod 驱动 ENV_SLOTS 路并行 episode：批量 predict → env step → 组样本 |
| `kaiwu_env`（zmq server 环境进程，端口 5566） | `kaiwu_sts2.env_server` | REP socket + msgpack，K 个 env slot 复用一个进程；**底层是 2.7MB NativeAOT 引擎，不是真游戏** |
| `actor`（remote_actor_predict，独立推理） | `kaiwu_sts2.actor` | zmq REP 批量推理，回传 action/logp/value；定时从 modelpool 拉新模型热替换 |
| `learner`（on-policy：zmq PUSH→learner_server:9998） | `kaiwu_sts2.learner` | PULL 收整段 episode → GAE → masked PPO 更新 → 打包 zip 推 modelpool |
| `modelpool`（Go 二进制，HTTP :10014） | `kaiwu_sts2.modelpool` | ~60 行 stdlib HTTP：`/latest` `/upload?version=N` `/download?version=N` |
| 配置：`conf/kaiwudrl/*.toml` + sed 注入 | `conf/configure.toml` + 环境变量 | tomllib 读取，env 覆盖（容器里不再需要 sed 改写配置） |
| 一容器一角色：`utils/start.sh <role>` | `docker/compose.yaml` + `ROLE` env | 同一镜像，ENTRYPOINT 按 ROLE 分发 |
| msgpack 序列化 / zmq over TCP | `kaiwu_sts2.codec` (msgpack+numpy) | 一致 |
| reverb（off-policy） | **省略** | 我们只用 on-policy 路径（zmq PUSH 直发 learner） |
| Horovod 多 learner / self-play / Elo | **省略** | 单 policy 单机；需要时再加 |

## 数据流

```
[aisrv pod × N]                 [learner]              [modelpool :10014]
  env_server(127.0.0.1:5566)      ^                     ^
  aisrv ──PUSH samples:9998───────┘ zip HTTP upload      │
     │                                                   │
     └──REQ predict:8888──> [actor] ──poll /download─────┘
```

观测约定：喂给网络的 obs = 原始观测(199 维) 拼接 action mask(32 维)；
mask 由 `MaskedMLP.forward` 从 obs 尾部取出做 logits 屏蔽。

## 使用

```bash
# 构建镜像（workspace 根目录）
docker build -f docker/Dockerfile -t sts2-rl:latest .

# 环境自检（不需要集群）
docker run --rm -e ROLE=smoke sts2-rl:latest

# 起集群（默认 1 learner + 1 actor + 3 aisrv pod，全 headless）
docker compose -f docker/compose.yaml up
# 扩 rollout 端：
docker compose -f docker/compose.yaml up --scale aisrv=8

# 不用 Docker 的本机联调（Windows 直接用 win-x64 引擎跑五进程）
venv/Scripts/python.exe bench/run_local_cluster.py 90
```

## 实测（本机 16 核，Windows 宿主五进程联调）

- aisrv 单 pod（4 slot）：rollout **~1035 steps/s**（整局 env，reset 流水线化后；优化前 ~600）
- reset 流水线：`env_server` 每 slot 双缓冲，后台 3 线程预取下一个 seed 的初始状态
  （45ms 地图生成移出关键路径，critical path 只剩 O(μs) 换岗）；确定性已验证
  （流水线重置 == 显式同 seed 重置，逐帧一致）
- learner：BATCH_SIZE=4096 时约 500 train steps/s；模型版本经 modelpool 热替换到 actor（日志可见 v3→v7→v9）
- 训练吞吐瓶颈在 learner（CPU torch），与单机 SB3 测量一致（~1200 fps 上限量级）

## 与单机 SB3 路线的关系

`emulator/scripts/train.py`（SB3 MaskablePPO + SubprocVecEnv）仍可用于快速实验；
kaiwu_sts2 集群用于正式训练（learner/actor 资源隔离、rollout 独立扩容、
模型版本化管理）。两者共享同一个 NativeAOT 引擎与观测约定。

## 遗留

- 真游戏 parity（STS2MCP trace 对比）等游戏存档空闲后进行
- learner 的 approx_kl 偶发尖峰（2+），正式训练时考虑 LR schedule / KL early stop
- `ENV_KIND=combat` 路径（11 万 steps/s/核）尚未在集群里压测
