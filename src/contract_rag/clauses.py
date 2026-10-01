"""The clause types this project looks for.

Everything else (loading labels, evaluation, reports) reads this list, so adding a
clause means adding one entry here. ``cuad_category`` must match the CUAD name exactly.
"""

from pydantic import BaseModel


class ClauseSpec(BaseModel):
    key: str  # short id used in code and reports
    cuad_category: str  # name of the category in CUAD
    questions: list[str]  # how a user might ask about this clause


CLAUSES: list[ClauseSpec] = [
    ClauseSpec(
        key="governing_law",
        cuad_category="Governing Law",
        questions=[
            "Which law governs this agreement?",
            "Which country or state's laws apply to this contract?",
        ],
    ),
    ClauseSpec(
        key="termination_for_convenience",
        cuad_category="Termination For Convenience",
        questions=[
            "Can either party terminate this agreement without cause?",
            "Can the agreement be ended early for any reason, without a breach?",
        ],
    ),
    ClauseSpec(
        key="renewal_term",
        cuad_category="Renewal Term",
        questions=[
            "Does the agreement renew automatically, and for how long?",
            "What happens when the initial term of the contract ends?",
        ],
    ),
    ClauseSpec(
        key="cap_on_liability",
        cuad_category="Cap On Liability",
        questions=[
            "Is there a cap on liability?",
            "What is the maximum amount a party can be liable for under this agreement?",
        ],
    ),
    ClauseSpec(
        key="uncapped_liability",
        cuad_category="Uncapped Liability",
        questions=[
            "Are there any liabilities that are not subject to the liability cap?",
            "Is any party's liability unlimited?",
        ],
    ),
    ClauseSpec(
        key="exclusivity",
        cuad_category="Exclusivity",
        questions=[
            "Does this agreement grant exclusive rights to any party?",
            "Is either party prevented from dealing with other parties?",
        ],
    ),
    ClauseSpec(
        key="non_compete",
        cuad_category="Non-Compete",
        questions=[
            "Is any party restricted from competing with the other party?",
            "Does the contract contain a non-compete restriction?",
        ],
    ),
    ClauseSpec(
        key="anti_assignment",
        cuad_category="Anti-Assignment",
        questions=[
            "Does assigning this agreement to someone else require consent?",
            "Can either party transfer its rights under this contract to a third party?",
        ],
    ),
]
