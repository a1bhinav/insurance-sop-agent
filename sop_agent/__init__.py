"""SOP-guided conversational agent harness for insurance claims support.

Architecture (one turn):

    user text ──► Extractor (LLM, structured JSON) ──► Memory.absorb()
                                                        │
                                                        ▼
                                           SOP Controller (deterministic)
                                   phases · gates · allowed actions · tools
                                                        │
                                                        ▼ Plan (directive + grounded facts)
                                   Responder (LLM, phase-constrained prompt)
                                                        │
                                                        ▼
                                          Guards (leak / PII filters) ──► reply

The LLM interprets and phrases; the SOP owns phase order, safety gates and
which facts may be spoken at all.
"""

__version__ = "1.0.0"
