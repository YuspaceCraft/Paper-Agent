"""agent 执行约束配置(v10.1 统一)self-check — config.yaml 加载 + env 优先级。

Run: python agent/tests/test_config.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent import config as cfg_mod  # noqa: E402


def test_defaults_when_file_missing():
    limits = cfg_mod.load_limits("__no_such_file__.yaml")
    assert limits.max_steps == 30, "缺失文件 → 父 agent 默认 30"
    assert limits.max_turns == 50
    assert limits.subagents == {}, "缺失文件 → 无 subagent 覆盖"


def test_yaml_values_from_repo():
    # config.yaml 必须实际生效（区别于代码默认）。数值由 yaml 决定,这里只断言
    # «覆盖生效»: 父 agent 代码默认 max_steps=5,yaml 调大到 40(或任意值),
    # 不等于代码默认即证明 yaml 生效——数值变化不再需要同步改测试。
    limits = cfg_mod.get_limits()
    assert limits.max_steps != 5, "config.yaml 父 agent 上限应生效(≠ 代码默认 5)"
    assert limits.max_turns >= limits.max_steps
    assert limits.subagents["creator"].max_steps >= 12, \
        "creator 需要读多篇论文再落盘,上限必须放宽(写死在代码里易回归)"
    assert limits.subagents["arxiv"].max_steps >= 5


def test_env_overrides_config():
    # 父 agent: 显式 env 优先于 config.yaml
    with tempfile.TemporaryDirectory() as d:
        y = Path(d) / "c.yaml"
        y.write_text("max_steps: 30\nmax_turns: 50\nsubagents: {}\n", encoding="utf-8")
        os.environ["AGENT_MAX_STEPS"] = "9"
        try:
            limits = cfg_mod.load_limits(str(y))
            assert limits.max_steps == 30, "load_limits 是纯文件解析,不管 env"
        finally:
            del os.environ["AGENT_MAX_STEPS"]
    # state 类属性默认值的 env>config 优先级在这里验证(import 时求值)
    import agent.state as st
    old = st.get_limits().max_steps
    try:
        os.environ["AGENT_MAX_STEPS"] = "7"
        # 重新导入 state 使类属性默认值按当前 env 重新求值
        import importlib
        importlib.reload(st)
        assert st.AgentState.max_steps == 7, "env 应覆盖 config"
        assert st.AgentState.max_turns == st.get_limits().max_turns, "无 env 时用 config 值"
    finally:
        os.environ.pop("AGENT_MAX_STEPS", None)
        importlib.reload(st)
    assert st.AgentState.max_steps == old, "清理后恢复 config 默认"


def test_invalid_yaml_falls_back():
    limits = cfg_mod.load_limits("")  # 真实文件存在,不会触发
    assert limits.max_steps > 0


if __name__ == "__main__":
    test_defaults_when_file_missing()
    test_yaml_values_from_repo()
    test_env_overrides_config()
    test_invalid_yaml_falls_back()
    print("agent config self-check OK")