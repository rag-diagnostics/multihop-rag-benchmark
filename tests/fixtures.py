"""Artificial records for tests only, not benchmark data or measured model results."""
def chain(i=0, family="both"):
    gold = [f"Arlen{i} was the envoy of Alderia{i}.", f"Alderia{i} was allied with Belvar{i}.",
            f"Belvar{i} bordered Corven{i}.", f"Corven{i} had capital Doran{i}."]
    questions = [f"Which envoy served Alderia{i}?", f"Which envoy served the nation allied with Belvar{i}?",
        f"Which envoy served the nation allied with the nation bordering Corven{i}?",
        f"Which envoy served the nation allied with the nation bordering the nation whose capital is Doran{i}?"]
    bridges = [f"Alderia{i}", f"Belvar{i}", f"Corven{i}"]
    rows = {"original_question": questions[0], "answer": f"Arlen{i}", "decomposition": [
        {"hop_level": n+2, "entity_to_resolve": b, "clue_description": "synthetic clue", "reasoning": "test only"} for n,b in enumerate(bridges)], "hops": {}}
    for n in range(1,5):
        h = {"question": questions[n-1], "golden_docs" if n==1 else "all_golden_docs": gold[:n]}
        if n>1: h["bridge_entity_replaced"] = bridges[n-2]
        for f in (["relevant","irrelevant"] if family=="both" else ["random"]):
            prefix = "context_distractors" if f=="random" else "context_"+f
            h[prefix+("_remaining" if n>1 else "")] = [f"{f} report {j} in chain {i}. " + ("Independent news. " * (j+1)) for j in range(11-n)]
        rows["hops"][f"n{n}"]=h
    return rows
