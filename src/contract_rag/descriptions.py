"""One plain-language definition per clause type, used in the prompt for the LLM.

They are written in simple words, based on the official CUAD category descriptions.
Keys are the clause keys from clauses.py.
"""

CLAUSE_DEFINITIONS: dict[str, str] = {
    "governing_law": (
        "States which country's or state's law governs the contract and how it is interpreted."
    ),
    "termination_for_convenience": (
        "Lets a party end the contract early without cause (no breach needed), "
        "usually by giving notice."
    ),
    "renewal_term": (
        "Says what happens when the initial term ends: automatic renewal, extension, "
        "or renewal periods and how long they last."
    ),
    "cap_on_liability": (
        "Limits how much a party can be required to pay the other under the contract, "
        "or limits the time in which claims can be brought."
    ),
    "uncapped_liability": (
        "Says that a party's liability is NOT limited in some situations, for example "
        "indemnities, breach of confidentiality, gross negligence or willful misconduct "
        "are excluded from the liability cap."
    ),
    "exclusivity": (
        "An exclusive dealing commitment: exclusive rights or licenses, a promise to buy "
        "or sell only from or to one party, or a ban on dealing with third parties."
    ),
    "non_compete": (
        "Restricts a party from competing with the other party, or from operating in a "
        "certain market, region or line of business."
    ),
    "anti_assignment": (
        "Requires consent or notice before a party assigns or transfers the contract "
        "or its rights to a third party."
    ),
}
