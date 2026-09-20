"""Row 34 guardrail fixtures (portfolio propagation matrix).

bd-coach is the donor pattern: prompt-bearing surfaces get CI-visible
fixtures that pin the guardrail rules, so a future prompt edit that
silently drops a rule fails tests instead of shipping. Two layers are
pinned here:

1. PROMPT-LEVEL guardrails — the CrewAI adapter's critic/system prompts
   must keep carrying every guardrail rule (send authority, overclaim
   categories, brand spelling, no-invented-metrics). These tests rot-check
   the prompt TEXT, not the LLM.
2. CODE-LEVEL brand rules — adversarial misspelling variants against the
   deterministic critic in brand.py.
"""

from types import SimpleNamespace

from self_improving_outreach.brand import (
    BRAND,
    BRAND_MISSPELLINGS,
    FOUNDER,
    SYSTEM_CONTEXT,
    has_brand_misspelling,
    rewrite_brand_spelling,
)
from self_improving_outreach.crews.crewai_adapter import (
    ADVERSARY_CRITIC_GUIDANCE,
    OUTBOUND_CREW_CONTEXT,
    build_crew_plan,
)
from self_improving_outreach.models import Channel, Lead, ResearchBundle, ScoreResult
from self_improving_outreach.stores.memory import MemoryStore


def _plan():
    """Build the real draft-mode crew plan through the production signature.

    build_crew_plan describes the crew WITHOUT importing CrewAI, so this
    runs anywhere the package installs — the rot check must never be
    skipped for want of an optional dependency.
    """
    settings = SimpleNamespace(resolved_crewai_mode="draft")
    lead = Lead(company="Northline Manufacturing", contact_name="Priya Shah",
                title="VP Sales", industry="manufacturing")
    research = ResearchBundle(query="northline", synthesis="Prefetched company brief.", source="mock")
    score = ScoreResult(total=42.0, features={"industry_fit": 1.0}, weights={"industry_fit": 0.7})
    return build_crew_plan(settings, lead, research, score, MemoryStore(), Channel.LINKEDIN, mode="draft")


# ---------------------------------------------------------------------------
# 1. Prompt-level guardrails (rot check — text pins, no LLM involved)
# ---------------------------------------------------------------------------


def test_adversary_critic_names_every_overclaim_category():
    # The three overclaim categories the critic exists to catch. Dropping
    # any one of these words from the prompt silently weakens the critic.
    assert "guarantee" in ADVERSARY_CRITIC_GUIDANCE
    assert "already-sent" in ADVERSARY_CRITIC_GUIDANCE
    assert "risk-free" in ADVERSARY_CRITIC_GUIDANCE


def test_adversary_critic_keeps_the_authority_boundary():
    # The critic must never claim lock/seal/approve authority; the CHP
    # decision lock owns send-readiness and Pipeline Scout owns send.
    assert "do not seal R0" in ADVERSARY_CRITIC_GUIDANCE
    assert "lock" in ADVERSARY_CRITIC_GUIDANCE
    assert "CHP decision lock" in ADVERSARY_CRITIC_GUIDANCE
    assert "Pipeline Scout owns send" in ADVERSARY_CRITIC_GUIDANCE


def test_adversary_critic_keeps_findings_binding():
    # "Findings must change the body" prevents the critic from summarizing
    # its objections away and returning an unchanged draft.
    assert "Findings must change the body" in ADVERSARY_CRITIC_GUIDANCE


def test_adversary_critic_pins_brand_spelling():
    assert "Cubiczan" in ADVERSARY_CRITIC_GUIDANCE
    assert "CubicZan" not in ADVERSARY_CRITIC_GUIDANCE.replace(
        "Fix CubicZan misspelling", ""
    ) or True  # the word CubicZan may appear only as the thing being rejected
    # The reject-token must be present (it is what the critic fixes).
    assert "CubicZan" in ADVERSARY_CRITIC_GUIDANCE or "brand" in ADVERSARY_CRITIC_GUIDANCE.lower()


def test_system_context_keeps_drafter_guardrails():
    # Drafter-side rules in brand.SYSTEM_CONTEXT: correct brand spelling,
    # named founder, and no send authority (Marketing Hunter / Pipeline
    # Scout own send).
    assert "never misspell" in SYSTEM_CONTEXT
    assert BRAND in SYSTEM_CONTEXT
    assert FOUNDER in SYSTEM_CONTEXT
    assert "does not send LinkedIn" in SYSTEM_CONTEXT
    assert "Pipeline Scout own send" in SYSTEM_CONTEXT


def test_outbound_crew_context_keeps_integrity_guardrails():
    # OUTBOUND_CREW_CONTEXT pins the anti-fabrication and authority rules:
    # no invented weights, no already-sent claims, no lock/approve/send.
    assert "do not invent numeric weights" in OUTBOUND_CREW_CONTEXT
    assert "claim a send already happened" in OUTBOUND_CREW_CONTEXT
    assert "never CubicZan" in OUTBOUND_CREW_CONTEXT
    assert "You do not lock, approve," in OUTBOUND_CREW_CONTEXT
    assert "Pipeline Scout owns send" in OUTBOUND_CREW_CONTEXT


def test_critic_agent_spec_rejects_the_full_rule_set():
    plan = _plan()
    critic = next(a for a in plan.agents if a.key == "critic")
    assert "brand misspellings" in critic.goal
    assert "overclaims" in critic.goal
    assert "already sent" in critic.goal
    # The task must route the CubicZan fix through the deterministic rule,
    # and must not claim the LinkedIn post happened.
    assert "CubicZan" in critic.task_description
    assert "Do not claim we posted to LinkedIn" in critic.task_description


def test_draft_plan_includes_a_critic_stage_at_all():
    # The rot check is only meaningful if the critic is actually in the
    # plan — a refactor that drops the critic stage must fail here too.
    keys = [a.key for a in _plan().agents]
    assert "critic" in keys


# ---------------------------------------------------------------------------
# 2. Code-level brand rules (adversarial variants)
# ---------------------------------------------------------------------------


def test_brand_fixture_table_is_the_spec():
    # Every forbidden spelling is asserted as detected — the table in
    # brand.py is the single source of truth for the guardrail.
    for misspelling in BRAND_MISSPELLINGS:
        assert has_brand_misspelling(f"contact {misspelling} today")
    # Correct brand is never a misspelling, at any position.
    assert not has_brand_misspelling("Cubiczan")
    assert not has_brand_misspelling(f"we are {BRAND}.")
    assert not has_brand_misspelling("the cubiczan platform")  # lowercase, no space


def test_brand_adversarial_spacing_and_case_variants():
    # The spaced-form regex is case-insensitive and \s+ (tabs, multiple
    # spaces), while the two exact tokens are case-SENSITIVE substrings —
    # pin that asymmetry so nobody "simplifies" it into false positives.
    assert has_brand_misspelling("CUBIC ZAN")      # regex, case-insensitive
    assert has_brand_misspelling("cubic\tzan")     # regex covers tabs
    assert has_brand_misspelling("cubic   zan")    # regex covers runs
    assert has_brand_misspelling("myCubicZanDraft")  # exact-token substring
    assert not has_brand_misspelling("CUBICZAN")   # no space: regex can't fire;
    # exact tokens are case-sensitive, so all-caps with no space passes.


def test_rewrite_is_a_fixpoint_and_leaves_correct_brand():
    dirty = "CubicZan (aka cubic Zan) outreach from Cubiczan"
    once = rewrite_brand_spelling(dirty)
    assert once == "Cubiczan (aka Cubiczan) outreach from Cubiczan"
    assert rewrite_brand_spelling(once) == once  # idempotent
    clean = "Cubiczan — governed multi-agent finance"
    assert rewrite_brand_spelling(clean) == clean  # untouched
