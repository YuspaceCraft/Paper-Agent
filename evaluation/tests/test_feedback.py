import asyncio

from evaluation.feedback import submit_feedback


def test_submit_feedback_uses_explicit_run():
    calls = []

    class Client:
        def create_feedback(self, **kwargs):
            calls.append(kwargs)

    result = asyncio.run(submit_feedback(
        trace_id="trace-1",
        run_id="run-1",
        key="human_acceptance",
        score=1,
        comment="ok",
        client=Client(),
    ))

    assert result["outcome"] == "succeeded"
    assert calls == [{
        "run_id": "run-1",
        "key": "human_acceptance",
        "score": 1,
        "comment": "ok",
    }]
