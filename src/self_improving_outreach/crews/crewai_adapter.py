"""CrewAI adapter. Optional — mock/deterministic pipeline does not import this unless live.

Current CrewAI (docs v1.15): `from crewai import Agent, Task, Crew, Process`
Custom tools subclass `crewai.tools.BaseTool`.

CrewAI writes prose inside a swarm worker. It does not own ICP Score math,
Learner updates, the code brand critic, the CHP decision lock, or send.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from self_improving_outreach.config import Settings
from self_improving_outreach.crews.pipeline import draft_message
from self_improving_outreach.llm import apply_llm_runtime_env, build_crewai_llm
from self_improving_outreach.models import Channel, Draft, Lead, ResearchBundle, ScoreResult
from self_improving_outreach.observability.tracing import RunTracer
from self_improving_outreach.stores.base import OutreachStore

TOOL_YOU = "you_com_search"
TOOL_MEMORY = "outreach_memory_read"

YouRefresh = Callable[[str], str]

CREWAI_ADVERSARY_NOTE = "crewai.adversary_critic"

OUTBOUND_CREW_CONTEXT = (
    "You are part of Self Improving Outreach (Cubiczan). This product drafts "
    "outbound sales notes for any ICP — not only finance or CFO/CIO titles. "
    "Stored message_patterns and icp_weights are learner-updated examples; "
    "do not invent numeric weights or claim a send already happened. "
    "Brand spelling is Cubiczan (never CubicZan). You do not lock, approve, "
    "or send. The CHP decision lock owns send-readiness. Pipeline Scout owns send."
)

ADVERSARY_CRITIC_GUIDANCE = (
    "You are a CrewAI draft-brain adversary critic. You harden prose only. "
    "You do not seal R0, skip the structural adversary, lock, or approve for "
    "scout. The Cubiczan CHP decision lock (R0 → structural adversary → named "
    "human lock → evidence pack) is authoritative for send-readiness. Argue "
    "against weak claims, invented metrics, compliance and send risks, and "
    "overclaims (guarantee, already-sent, risk-free). Findings must change "
    "the body, not be summarized away. Return the hardened draft body only. "
    "Brand spelling is Cubiczan (never CubicZan). Pipeline Scout owns send."
)


def crewai_available() -> bool:
    try:
        import crewai  # noqa: F401

        return True
    except Exception:
        return False


def optional_chp_foundation_floor(domain: str = "general") -> Optional[int]:
    """Soft import of an optional external CHP floor. Not the in-repo lock."""
    try:
        import consensus_hardening_protocol as module  # type: ignore
    except Exception:
        return None
    foundation = getattr(module, "foundation", module)
    for name in (
        "resolve_floor",
        "resolve_foundation_floor",
        "foundation_floor",
        "floor_for_domain",
        "floor",
    ):
        fn = getattr(foundation, name, None)
        if not callable(fn):
            continue
        try:
            return int(fn(domain))
        except Exception:
            continue
    return None


def adversary_prompt_suffix() -> str:
    extra = ""
    floor = optional_chp_foundation_floor("general")
    if floor is not None:
        extra = (
            f" Optional consensus-hardening-protocol foundation floor for "
            f"general outbound is {floor}. This is not the Cubiczan CHP lock."
        )
    return ADVERSARY_CRITIC_GUIDANCE + extra


def _pattern_blob(store: OutreachStore, limit: int = 8) -> str:
    patterns = store.list_patterns()
    if not patterns:
        return "(no stored patterns)"
    return "\n".join(f"- {p.pattern_id} ({p.score:.2f}): {p.angle}" for p in patterns[:limit])


def _count_tool(counter: Optional[dict[str, int]], name: str) -> None:
    if counter is None:
        return
    counter[name] = counter.get(name, 0) + 1


def build_you_tool(you_refresh: YouRefresh, counter: Optional[dict[str, int]] = None) -> Any:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class SearchInput(BaseModel):
        query: str = Field(..., description="Company / contact / market research query")

    class YouComSearchTool(BaseTool):
        name: str = TOOL_YOU
        description: str = (
            "Search You.com for live company, contact, and market context for "
            "general outbound. Retries once then degrades to cached context."
        )
        args_schema: type[BaseModel] = SearchInput

        def _run(self, query: str) -> str:
            _count_tool(counter, TOOL_YOU)
            return you_refresh(query)

    return YouComSearchTool()


def build_memory_tool(store: OutreachStore, counter: Optional[dict[str, int]] = None) -> Any:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class MemoryInput(BaseModel):
        kind: str = Field(
            ...,
            description="Read 'patterns' (scored message_patterns) or 'weights' (icp_weights).",
        )

    class OutreachMemoryTool(BaseTool):
        name: str = TOOL_MEMORY
        description: str = (
            "Read-only scored message_patterns or ICP weights from the store. "
            "Do not invent weights. Use these as examples and pick among them."
        )
        args_schema: type[BaseModel] = MemoryInput

        def _run(self, kind: str) -> str:
            _count_tool(counter, TOOL_MEMORY)
            requested = (kind or "patterns").strip().lower()
            if requested.startswith("weight"):
                weights = store.get_weights()
                if not weights:
                    return "(no stored weights)"
                return "\n".join(f"{key}={value}" for key, value in weights.items())
            return _pattern_blob(store, limit=20)

    return OutreachMemoryTool()


@dataclass(frozen=True)
class CrewAgentSpec:
    key: str
    role: str
    goal: str
    backstory: str
    tool_names: tuple[str, ...]
    task_description: str
    expected_output: str


@dataclass(frozen=True)
class CrewPlan:
    mode: str
    agents: tuple[CrewAgentSpec, ...]

    @property
    def tool_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for agent in self.agents:
            for name in agent.tool_names:
                if name not in names:
                    names.append(name)
        return tuple(names)

    def agent(self, key: str) -> CrewAgentSpec:
        for spec in self.agents:
            if spec.key == key:
                return spec
        raise KeyError(key)


def build_crew_plan(
    settings: Settings,
    lead: Lead,
    research: ResearchBundle,
    score: ScoreResult,
    store: OutreachStore,
    channel: Channel,
    *,
    mode: Optional[str] = None,
) -> CrewPlan:
    """Describe the sequential crew without importing CrewAI."""
    resolved = mode or settings.resolved_crewai_mode
    if resolved == "off":
        return CrewPlan(mode="off", agents=())

    contact = lead.contact_name or "there"
    pattern_blob = _pattern_blob(store)
    research_text = research.as_text() or "(empty research bundle)"
    shared = OUTBOUND_CREW_CONTEXT

    researcher = CrewAgentSpec(
        key="researcher",
        role="Outbound Researcher",
        goal="Build a concise company and contact brief for general outbound",
        backstory=shared + " You use live search when available and never invent filings or metrics.",
        tool_names=(TOOL_YOU,) if resolved == "full" else (),
        task_description=(
            f"Brief {lead.company} / {contact} ({lead.title or 'contact'}) for outbound. "
            f"Industry={lead.industry or 'unspecified'}. Pre-fetched research:\n{research_text}\n"
            "If you have a search tool, refresh facts that look stale or thin. "
            "Do not invent quotes, filings, or numbers."
        ),
        expected_output="A 5-bullet company/contact brief for outbound.",
    )
    scorer = CrewAgentSpec(
        key="scorer",
        role="Interpreter Scorer",
        goal="Explain the deterministic ICP score already computed from stored weights",
        backstory="You do not invent or recompute weights. You interpret the numeric score.",
        tool_names=(),
        task_description=(
            f"ICP score is {score.total}. Features={score.features}. "
            f"Weights={score.weights}. In two sentences, explain fit for this outbound ICP. "
            "Do not invent a new total."
        ),
        expected_output="Two-sentence ICP rationale that does not change the numeric score.",
    )
    drafter = CrewAgentSpec(
        key="drafter",
        role="Outbound Drafter",
        goal=f"Write a short {channel.value} outbound note using a scored pattern angle",
        backstory=shared,
        tool_names=(),
        task_description=(
            f"Write a {channel.value} note to {contact} at {lead.company}. "
            f"Prefer these scored patterns (examples, not the only possible ICP):\n{pattern_blob}\n"
            "Do not say we already sent it. Brand spelling is Cubiczan if the brand appears."
        ),
        expected_output="A short outreach draft body only.",
    )

    if resolved == "draft":
        critic = CrewAgentSpec(
            key="critic",
            role="Outbound Critic",
            goal="Reject brand misspellings, overclaims, and language that implies we already sent",
            backstory=(
                "Pipeline Scout owns send. The CHP lock owns send-readiness. "
                "You only polish the draft body."
            ),
            tool_names=(),
            task_description=(
                "Critique and return the final draft body only. "
                "Fix CubicZan misspelling if present. Do not claim we posted to LinkedIn."
            ),
            expected_output="Final draft body.",
        )
        return CrewPlan(mode="draft", agents=(researcher, scorer, drafter, critic))

    strategist = CrewAgentSpec(
        key="strategist",
        role="Outbound Strategist",
        goal="Pick and justify one scored message_pattern angle for this lead",
        backstory=(
            shared + " You choose among stored patterns. You do not invent weights "
            "or new pattern scores."
        ),
        tool_names=(TOOL_YOU, TOOL_MEMORY),
        task_description=(
            f"Choose one stored pattern for {contact} at {lead.company}. "
            f"Scored patterns:\n{pattern_blob}\n"
            "Justify the angle from the brief and the deterministic score. "
            "If you have memory/search tools, re-read patterns or refresh a fact. "
            "Do not invent a weight or a new pattern_id."
        ),
        expected_output="Chosen pattern_id, angle, and a 2-sentence justification.",
    )
    adversary = CrewAgentSpec(
        key="critic",
        role="Adversary Critic",
        goal="Attack weak claims, compliance/send risks, and overclaims; return a hardened body",
        backstory=adversary_prompt_suffix(),
        tool_names=(),
        task_description=(
            f"{adversary_prompt_suffix()} Channel is {channel.value}. "
            "Return only the hardened final body."
        ),
        expected_output="Hardened final draft body only.",
    )
    return CrewPlan(
        mode="full",
        agents=(researcher, scorer, strategist, drafter, adversary),
    )


def _default_you_refresh(research: ResearchBundle, store: OutreachStore, lead: Lead) -> YouRefresh:
    def _refresh(query: str) -> str:
        cached = store.cached_context(lead)
        return research.as_text() or cached or f"No live refresh available for: {query}"

    return _refresh


def _emit(tracer: Optional[RunTracer], name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    if tracer is None:
        return
    tracer.event(name, attributes or {})


def run_crewai_draft(
    settings: Settings,
    lead: Lead,
    research: ResearchBundle,
    score: ScoreResult,
    store: OutreachStore,
    channel: Channel,
    *,
    tracer: Optional[RunTracer] = None,
    you_refresh: Optional[YouRefresh] = None,
) -> Optional[Draft]:
    mode = settings.resolved_crewai_mode
    _emit(tracer, "crewai.mode", {"mode": mode})
    if mode == "off":
        return None
    if not crewai_available():
        _emit(tracer, "crewai.fallback", {"reason": "unavailable"})
        return None

    from crewai import Agent, Crew, Process, Task

    apply_llm_runtime_env(settings)
    llm = build_crewai_llm(settings)
    agent_kwargs: dict[str, Any] = {
        "verbose": settings.crewai_verbose,
        "allow_delegation": False,
    }
    if llm is not None:
        agent_kwargs["llm"] = llm

    plan = build_crew_plan(settings, lead, research, score, store, channel, mode=mode)
    counter: dict[str, int] = {}
    refresh = you_refresh or _default_you_refresh(research, store, lead)
    built_tools: dict[str, Any] = {}
    if TOOL_YOU in plan.tool_names:
        built_tools[TOOL_YOU] = build_you_tool(refresh, counter)
    if TOOL_MEMORY in plan.tool_names:
        built_tools[TOOL_MEMORY] = build_memory_tool(store, counter)

    _emit(
        tracer,
        "crewai.tools",
        {
            "attached": len(plan.tool_names),
            "calls": 0,
            "names": list(plan.tool_names),
        },
    )

    materialized: dict[str, Any] = {}
    tasks: list[Any] = []
    previous: Optional[Any] = None
    for spec in plan.agents:
        tools = [built_tools[name] for name in spec.tool_names if name in built_tools]
        agent = Agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            tools=tools,
            **agent_kwargs,
        )
        materialized[spec.key] = agent
        task_kwargs: dict[str, Any] = {
            "description": spec.task_description,
            "expected_output": spec.expected_output,
            "agent": agent,
        }
        if previous is not None:
            task_kwargs["context"] = [previous]
        task = Task(**task_kwargs)
        tasks.append(task)
        previous = task

    fallback = draft_message(lead, store, channel)
    try:
        crew = Crew(
            agents=list(materialized.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=settings.crewai_verbose,
        )
        result = crew.kickoff()
    except Exception as exc:  # noqa: BLE001 — live LLM/tool failures must not kill the unit
        _emit(tracer, "crewai.fallback", {"reason": "kickoff", "error": str(exc)})
        return fallback

    _emit(
        tracer,
        "crewai.tools",
        {
            "attached": len(plan.tool_names),
            "calls": int(sum(counter.values())),
            "names": list(plan.tool_names),
        },
    )
    body = str(getattr(result, "raw", result)).strip()
    if not body:
        _emit(tracer, "crewai.fallback", {"reason": "empty_body"})
        return fallback
    fallback.body = body
    if mode == "full":
        fallback.adversary_notes = [CREWAI_ADVERSARY_NOTE]
    return fallback
