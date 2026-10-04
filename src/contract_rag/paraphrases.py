"""A third set of queries: how a layperson might ask, in everyday words.

These are a stress test for keyword search. They deliberately avoid the vocabulary that
contracts use ("terminate", "assign", "governed by", ...), so a keyword match is hard and
the meaning has to carry the search. They are NOT used to tune anything.

Keys are the clause keys from clauses.py.
"""

PARAPHRASES: dict[str, list[str]] = {
    "governing_law": [
        "If we ever end up in a legal fight over this deal, whose legal system decides it?",
        "Which place's legal rules does this deal follow?",
    ],
    "termination_for_convenience": [
        "Can I walk away from this deal early without giving a reason?",
        "Is there a way out of this contract before it ends if I simply change my mind?",
    ],
    "renewal_term": [
        "Does this deal keep going by itself after the first period, or does it just stop?",
        "What happens at the end of the first period, does it roll over?",
    ],
    "cap_on_liability": [
        "What is the most money one side could be forced to pay the other if something goes wrong?",
        "Is there a ceiling on how much someone can be sued for under this deal?",
    ],
    "uncapped_liability": [
        "Are there situations where one side's financial responsibility has no upper limit?",
        "Which kinds of problems are not covered by the limit on damages?",
    ],
    "exclusivity": [
        "Is one side the only one allowed to sell, use or get this product or service?",
        "Is anyone locked in to dealing only with the other party?",
    ],
    "non_compete": [
        "Is either side forbidden from doing business with the other side's rivals?",
        "Can a party start or work in a similar business while this deal lasts?",
    ],
    "anti_assignment": [
        "Can the other side hand this contract over to somebody else without asking me?",
        "Do I need permission before passing my rights under this contract to another company?",
    ],
}
