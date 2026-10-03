"""The clause types this project looks for.

Everything else (loading labels, evaluation, reports) reads this list, so adding a
clause means adding one entry here. ``cuad_category`` must match the CUAD name exactly.

Two sets of queries per clause:
- ``questions``: how a user might ask ("v1", our baseline)
- ``questions_v2``: written like the contract text itself, based on the official
  CUAD category descriptions
"""

from pydantic import BaseModel


class ClauseSpec(BaseModel):
    key: str
    cuad_category: str
    questions: list[str]
    questions_v2: list[str] = []


CLAUSES: list[ClauseSpec] = [
    ClauseSpec(
        key="governing_law",
        cuad_category="Governing Law",
        questions=[
            "Which law governs this agreement?",
            "Which country or state's laws apply to this contract?",
        ],
        questions_v2=[
            "This Agreement shall be governed by and construed in accordance with the laws "
            "of the State of New York.",
            "The validity, interpretation and performance of this Agreement shall be governed "
            "by the laws of England and Wales.",
        ],
    ),
    ClauseSpec(
        key="termination_for_convenience",
        cuad_category="Termination For Convenience",
        questions=[
            "Can either party terminate this agreement without cause?",
            "Can the agreement be ended early for any reason, without a breach?",
        ],
        questions_v2=[
            "Either party may terminate this Agreement at any time, without cause, upon thirty "
            "days prior written notice to the other party.",
            "Customer may terminate this Agreement for any reason or no reason by giving "
            "written notice, effective when the notice period expires.",
        ],
    ),
    ClauseSpec(
        key="renewal_term",
        cuad_category="Renewal Term",
        questions=[
            "Does the agreement renew automatically, and for how long?",
            "What happens when the initial term of the contract ends?",
        ],
        questions_v2=[
            "This Agreement shall automatically renew for successive one-year terms unless "
            "either party gives written notice of non-renewal before the end of the then-current "
            "term.",
            "Upon expiration of the initial term, the term may be extended for additional "
            "renewal periods by written notice from a party.",
        ],
    ),
    ClauseSpec(
        key="cap_on_liability",
        cuad_category="Cap On Liability",
        questions=[
            "Is there a cap on liability?",
            "What is the maximum amount a party can be liable for under this agreement?",
        ],
        questions_v2=[
            "In no event shall either party's aggregate liability under this Agreement exceed "
            "the total fees paid by Customer in the twelve months preceding the claim.",
            "Any claim under this Agreement must be brought within one year, and the maximum "
            "amount recoverable shall not exceed the amounts paid hereunder.",
        ],
    ),
    ClauseSpec(
        key="uncapped_liability",
        cuad_category="Uncapped Liability",
        questions=[
            "Are there any liabilities that are not subject to the liability cap?",
            "Is any party's liability unlimited?",
        ],
        questions_v2=[
            "The limitations of liability in this Agreement shall not apply to indemnification "
            "obligations, breach of confidentiality, infringement of intellectual property, "
            "gross negligence or willful misconduct.",
            "Nothing in this Agreement limits or excludes either party's liability for fraud, "
            "death or personal injury, or damages caused by intentional misconduct.",
        ],
    ),
    ClauseSpec(
        key="exclusivity",
        cuad_category="Exclusivity",
        questions=[
            "Does this agreement grant exclusive rights to any party?",
            "Is either party prevented from dealing with other parties?",
        ],
        questions_v2=[
            "Licensor grants Licensee an exclusive right, and Licensor shall not grant any "
            "such rights to any third party.",
            "Customer shall purchase all of its requirements for the Products exclusively from "
            "Supplier and shall not purchase them from any other party.",
        ],
    ),
    ClauseSpec(
        key="non_compete",
        cuad_category="Non-Compete",
        questions=[
            "Is any party restricted from competing with the other party?",
            "Does the contract contain a non-compete restriction?",
        ],
        questions_v2=[
            "During the term, neither party shall, directly or indirectly, engage in any "
            "business that competes with the business of the other party.",
            "A party shall not sell or distribute any products that compete with the other "
            "party's products in the Territory.",
        ],
    ),
    ClauseSpec(
        key="anti_assignment",
        cuad_category="Anti-Assignment",
        questions=[
            "Does assigning this agreement to someone else require consent?",
            "Can either party transfer its rights under this contract to a third party?",
        ],
        questions_v2=[
            "Neither party may assign this Agreement or any rights hereunder without the prior "
            "written consent of the other party.",
            "Either party may assign this Agreement to an affiliate or successor only upon "
            "prior written notice to the other party.",
        ],
    ),
]
