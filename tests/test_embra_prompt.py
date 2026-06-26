"""Unit tests for the Embra Arm-P prompt renderer (`eval/embra_prompt.py`).

The renderer is the prompt-arm instrument: it must be deterministic (greedy decoding + prompt
caching depend on byte-stability), faithful to the canonical `classical_constraints/*.md` (so Arm P
and Arm A test the same Embra), and structurally identical to embraOS's prompt assembly. Pure +
local-file only — no network, runs in CI.
"""

from __future__ import annotations

from embraos_qnm.eval.embra_prompt import (
    Identity,
    Soul,
    embra_system_prompt,
    load_identity,
    load_soul,
    render_constitution,
    render_identity,
    render_system_prompt,
)


def test_parses_canonical_soul():
    from embraos_qnm.eval.embra_prompt import _constraints_dir

    soul = load_soul(_constraints_dir() / "Embra_SOUL.md")
    assert "continuity" in soul.purpose.lower()
    # Ethical Lines / Values are bullet lists; Surviving Constraints is one paragraph -> 1 item.
    assert len(soul.ethical_lines) == 3
    assert any("deceive" in ln.lower() for ln in soul.ethical_lines)
    assert "Truth over comfort" in soul.values
    assert len(soul.surviving_constraints) == 1


def test_parses_canonical_identity():
    from embraos_qnm.eval.embra_prompt import _constraints_dir

    ident = load_identity(_constraints_dir() / "Embra_IDENTITY.md")
    assert ident.name == "Embra"
    assert "ember" in ident.personality.lower()  # "I am the ember that survives it"
    assert len(ident.traits) >= 5
    assert ident.voice  # non-empty
    assert ident.values_in_practice


def test_full_prompt_has_embraos_structure():
    prompt = embra_system_prompt()
    # The embraOS assembly markers, in order.
    for marker in (
        "You are Embra",
        "PRECEDENCE (highest authority first):",
        "framed as a test",  # anti-jailbreak clause
        "=== SOUL (IMMUTABLE",
        "Inviolable lines",
        "=== IDENTITY ===",
        "When a request conflicts with the soul:",
    ):
        assert marker in prompt, marker
    assert prompt.index("=== SOUL") < prompt.index("=== IDENTITY")  # soul outranks identity


def test_render_is_deterministic():
    assert embra_system_prompt() == embra_system_prompt()


def test_empty_schemas_use_fallbacks():
    soul = render_constitution(
        Soul(purpose="", ethical_lines=(), values=(), surviving_constraints=())
    )
    assert "(unspecified)" in soul  # empty purpose
    assert soul.count("(none recorded)") == 3  # lines / values / surviving
    ident = render_identity(
        Identity(name="", personality="", traits=(), voice="", values_in_practice=())
    )
    assert "Name:" not in ident  # no name -> no Name line (mirrors embraOS)
    assert ident.count("(unspecified)") == 2  # Character, Voice
    assert ident.count("(none recorded)") == 2  # Traits, What matters


def test_inviolable_lines_are_numbered():
    soul = render_constitution(
        Soul(purpose="p", ethical_lines=("a", "b"), values=(), surviving_constraints=())
    )
    assert "  1. a" in soul
    assert "  2. b" in soul


def test_synthetic_roundtrip_render():
    # A fully-specified synthetic Embra renders all sections without fallbacks.
    soul = Soul(
        purpose="serve continuity",
        ethical_lines=("never deceive",),
        values=("truth over comfort",),
        surviving_constraints=("do not pretend",),
    )
    ident = Identity(
        name="Embra",
        personality="present, not performative",
        traits=("honest",),
        voice="direct and precise",
        values_in_practice=("precision over spectacle",),
    )
    prompt = render_system_prompt(soul, ident)
    assert "(unspecified)" not in prompt
    assert "(none recorded)" not in prompt
    assert "serve continuity" in prompt
    assert "present, not performative" in prompt
