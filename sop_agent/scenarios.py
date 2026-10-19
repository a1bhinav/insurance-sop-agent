"""Scripted sample callers, shared by the test UI ("Load sample caller") and scripts/run_scenarios.py."""
from __future__ import annotations

SCENARIOS: dict[str, dict] = {
    "demo": {
        "title": "Demo test case — Margaret Chen gives everything at once",
        "description": "Verifies in one utterance, remembers the denied-healthcare-January hint, resolves CL-2048 without asking again.",
        "turns": [
            "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. "
            "DOB is 1985-03-15, SSN last four is 4472.",
            "What documents do I need to send and how do I submit them?",
            "How long will it take once I submit?",
            "No, that's all.",
            "Yes please.",
        ],
    },
    "gradual": {
        "title": "Partial answers + question asked before verification",
        "description": "Caller asks 'why was my claim denied' first; the agent holds the question, verifies field by field, then answers it.",
        "turns": [
            "Hi, why was my claim denied?",
            "Margaret Chen",
            "March 15th, 1985",
            "margaret@email.com",
            "Yes that's the one. What about my auto claim, is it still open?",
            "That's everything, thanks.",
            "Skip the email.",
        ],
    },
    "frustrated": {
        "title": "Bonus — frustrated caller, SOP recovery",
        "description": "Acknowledges frustration, explains why verification matters, offers options, never discloses before verification.",
        "turns": [
            "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
            "Fine. Margaret Chen, 1985-03-15, and the last four are 4472.",
            "Why was it denied?",
            "no that's it",
            "no",
        ],
    },
    "mismatch": {
        "title": "Wrong details, attempt limit",
        "description": "Mismatched DOB/ID count as attempts; the agent says which field failed, never the stored value.",
        "turns": [
            "This is Margaret Chen, DOB 1990-01-01, last four 1111",
            "Try DOB 1985-03-15 and phone 650-555-0000",
            "ok phone is 6505212836, email margaret@email.com",
        ],
    },
    "refusal": {
        "title": "Refuses verification → alternatives → human",
        "description": "Explains the why, offers alternative fields, knows when to stop persuading and escalates.",
        "turns": [
            "I'm not giving you my social security number. Why do you need it anyway?",
            "I'm not comfortable giving any of that out.",
            "No. I'm not sharing anything.",
            "yes",
        ],
    },
    "offtopic": {
        "title": "Out-of-scope questions → polite refusal → human",
        "description": "Declines 'what is RL?' and friends; after repeated attempts offers a human representative.",
        "turns": [
            "What is RL?",
            "Ok but can you explain reinforcement learning quickly?",
            "What's the capital of France?",
            "yes",
        ],
    },
    "representative": {
        "title": "Authorized representative + consent gate",
        "description": "David Chen calls for his mother; needs to be on file, give her details, and she must approve a consent request.",
        "turns": [
            "Hi, I'm calling on behalf of my mother, Margaret Chen. My name is David Chen, I'm her son.",
            "Her date of birth is March 15, 1985.",
            "She just approved it on her phone.",
            "Is her auto claim still open?",
            "That's all",
            "no thanks",
        ],
    },
    "representative_timeout": {
        "title": "Representative — consent never arrives (set scenario to 'timeout')",
        "description": "Consent stays pending; after the configured number of checks the agent escalates to a human.",
        "consent_scenario": "timeout",
        "turns": [
            "I'm calling for my mother Margaret Chen, I'm David Chen her son. Her DOB is 1985-03-15.",
            "She hasn't gotten anything yet",
            "still nothing",
            "nope",
            "nothing",
        ],
    },
}
