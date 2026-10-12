"""Judge of RAGAS Faithfulness: an external LLM behind the RAGAS prompts.

The judge is Anthropic's API, reached through RAGAS' own statement and
verdict prompts, so a score here is the published metric and not a
re-implementation. Both steps are exposed separately because the stage
stores every statement and verdict, not only the score.
"""

from typing import Any, Protocol

import anthropic
import instructor
import ragas
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import Faithfulness

MAX_TOKENS = 8192
# A rate-limited CQ that gives up drops out of the score, so retry well past
# the SDK's default of 2.
MAX_RETRIES = 6


class Judge(Protocol):
    """What the faithfulness stage needs from a judge."""

    def describe(self) -> dict[str, str]:
        """Names the judge for the record of the cell."""
        ...

    async def statements(self, question: str, answer: str) -> list[str]:
        """Splits an answer into atomic statements."""
        ...

    async def verdicts(
        self, statements: list[str], context: str
    ) -> list[dict[str, Any]]:
        """Judges each statement against the context."""
        ...


class RagasJudge:
    """RAGAS Faithfulness with a Claude model as the judge.

    Attributes:
        model: The judge's model id.
    """

    def __init__(self, model: str, *, api_key: str) -> None:
        self.model = model
        # JSON mode: the tool-calling mode forces a tool choice, which the
        # current Claude models reject.
        client = instructor.from_anthropic(
            anthropic.AsyncAnthropic(api_key=api_key, max_retries=MAX_RETRIES),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )
        llm = InstructorLLM(
            client=client, model=model, provider="anthropic", max_tokens=MAX_TOKENS
        )
        # The current Claude models reject sampling parameters, so RAGAS'
        # default temperature is dropped; the stored verdicts are the record.
        for name in ("temperature", "top_p"):
            llm.model_args.pop(name, None)
        # The two steps below are private in RAGAS 0.4.3 (pinned): its public
        # ``ascore`` returns only the score, and the stage stores the verdicts.
        self._metric = Faithfulness(llm=llm)

    def describe(self) -> dict[str, str]:
        """Names the judge for the record of the cell."""
        return {
            "model": self.model,
            "ragas": ragas.__version__,
            "metric": "faithfulness",
            "max_tokens": str(MAX_TOKENS),
            "sampling": "model default (the model accepts no temperature)",
        }

    async def statements(self, question: str, answer: str) -> list[str]:
        """Splits an answer into atomic statements (RAGAS statement prompt)."""
        return list(await self._metric._create_statements(question, answer))

    async def verdicts(
        self, statements: list[str], context: str
    ) -> list[dict[str, Any]]:
        """Judges each statement against the context (RAGAS NLI prompt)."""
        output = await self._metric._create_verdicts(statements, context)
        bad = [s.verdict for s in output.statements if s.verdict not in (0, 1)]
        if bad:
            # The schema only says "int"; anything else would skew the score.
            raise ValueError(f"verdicts outside 0/1: {bad[:3]}")
        return [
            {"statement": s.statement, "verdict": int(s.verdict), "reason": s.reason}
            for s in output.statements
        ]
