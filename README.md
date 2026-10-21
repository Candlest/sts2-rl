# sts2-rl — 杀戮尖塔2 大规模 CPU RL 训练环境

Headless、小资源、大规模并行的 Slay the Spire 2 强化学习环境，容器化部署，
进程拓扑参考腾讯开悟（kaiwudrl）的 actor/learner 架构。详见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 架构速览

```
[aisrv pod × N]                 [learner]              [modelpool :10014]
  env_server(127.0.0.1:5566)      ^                     ^
  aisrv ──PUSH samples:9998───────┘ zip HTTP upload      │
     │                                                   │
     └──REQ predict:8888──> [actor] ──poll /download─────┘
```

- **训练引擎**：[Zamiell/slay-the-spire-2-emulator](https://github.com/Zamiell/slay-the-spire-2-emulator)
  的 NativeAOT C# 确定性引擎（2.7MB，无渲染/引擎依赖），Python ctypes 进程内调用
- **真游戏桥接**：[Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP) 游戏 mod，
  localhost REST :15526，只用于 parity 验证和最终评测
- **训练框架**：`kaiwu_sts2/`（本仓库）—— aisrv / env_server / actor / learner / modelpool 五角色
- **reset 流水线**：env_server 每 slot 双缓冲，后台线程预生成下一 seed 的地图（45ms 移出关键路径）

## 快速开始

```bash
# 1. 获取两个 MIT 协议的依赖（本仓库不包含它们）
git clone https://github.com/Zamiell/slay-the-spire-2-emulator.git emulator
git clone https://github.com/Gennadiyev/STS2MCP.git STS2MCP

# 2. 打上本仓库的跨平台补丁（Linux NativeAOT 符号导出修复）
git -C emulator apply ../patches/sts2emulator-linux-cross.patch

# 3. 构建训练镜像（含容器内 NativeAOT linux-x64 编译）
docker build -f docker/Dockerfile -t sts2-rl:latest .

# 4. 环境自检（headless）
docker run --rm -e ROLE=smoke sts2-rl:latest

# 5. 起训练集群（1 learner + 1 actor + 3 aisrv pod，全 headless）
docker compose -f docker/compose.yaml up
# 扩 rollout 端：docker compose -f docker/compose.yaml up --scale aisrv=8
```

Windows 宿主免 Docker 联调：

```bash
# 宿主直接构建 win-x64 引擎 + 五进程联合冒烟（需要 .NET 9 SDK 与 Python 3.13）
cd emulator && dotnet publish "src/Sts2Emulator/Sts2Emulator.csproj" -c Release -r win-x64 --self-contained -o out
cd .. && python -m venv venv
venv/Scripts/python.exe -m pip install -i https://mirrors.aliyun.com/pypi/simple/ gymnasium numpy pyzmq msgpack torch sb3-contrib stable-baselines3
venv/Scripts/python.exe bench/run_local_cluster.py 90
```

## 实测性能（16 核单机）

| 指标 | 数值 |
|---|---|
| 战斗 env 单核 step | 0.009 ms（~11 万 steps/s） |
| 整局 env 单核 step / reset | 1.64 ms / 45ms（reset 已流水线化，关键路径 ~0） |
| 模拟器确定性 | 同 seed 逐帧复现 ✅ |
| aisrv pod（4 slot，reset 流水线） | rollout ~1035 steps/s（优化前 ~600，+72%） |
| MaskablePPO 稳态（单机 SB3 对照） | ~1200 fps，瓶颈在 CPU learner |

## 关键事实

- 目标游戏版本 **v0.107.1**（所有 parity 工作锁版本；升级时重跑 emulator 的 `patch_update.sh` 管线）
- 游戏 RNG = Xoshiro256**（`MegaRandom`），seed + 每条 RNG 流调用计数即可完全复现一局
- 观测 = 199 维定长整数向量（战斗）+ 整局块；动作 32 维离散 + mask；网络输入为 obs 拼接 mask（231 维）
- 模拟器内容覆盖目前是 Act 1 / Ironclad 子集，扩展走反编译源码对照（本仓库不含反编译代码）

## 本机网络注意事项（中国大陆）

- PyPI 官方源超时、清华镜像 403 → 阿里云镜像：`pip install -i https://mirrors.aliyun.com/pypi/simple/`
- Dockerfile 已内置 apt/pip 阿里云源；torch CPU 走 `download.pytorch.org`（可用
  `--build-arg TORCH_INDEX=...` 覆盖）

## 合规

本仓库不含游戏二进制、资源或反编译代码。需要用户自行拥有 Slay the Spire 2
（Steam）副本；真游戏桥接 mod 只在本地运行、不修改游戏文件。依赖的
slay-the-spire-2-emulator 与 STS2MCP 均为 MIT 协议，由使用者自行获取。
