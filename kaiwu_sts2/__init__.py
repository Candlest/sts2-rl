"""Kaiwu-style actor-learner framework for Slay the Spire 2 CPU RL training.

Topology (mirrors kaiwudrl):
    [aisrv x N] --zmq PUSH samples--> [learner:9998]
    [aisrv x N] --zmq REQ predict ---> [actor:8888]
    [learner] --HTTP zip--> [modelpool:10014] <--poll-- [actor]
    [aisrv pod] = env_server(127.0.0.1:5566, K slots) + aisrv orchestrator

Simplifications vs. kaiwudrl: single policy, on-policy (no reverb), modelpool
is a ~60-line stdlib HTTP file pool instead of the Go binary.
"""

__version__ = "0.1.0"
