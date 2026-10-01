"""Versioned post-generation adapter for symmetric opposite-goal evidence.

The vendored PrOntoQA source is unchanged.  We remove its isolated one-edge
opposite-goal distractor and add a predicate-renamed copy of the connected
proof component.  The copy is grounded on a different entity, so its final
opposite conclusion is unreachable for the queried entity while local graph
statistics and proof-chain length are mirrored.
"""
from __future__ import annotations
import random

GENERATOR_VERSION = "prontoqa-symmetric-component-v2"

def literal(formula, fol):
    neg=isinstance(formula,fol.FOLNot); atom=formula.operand if neg else formula
    if not isinstance(atom,fol.FOLFuncApplication) or len(atom.args)!=1:return None
    return atom.function,neg

def rule(formula, fol):
    if not isinstance(formula,fol.FOLForAll) or not isinstance(formula.operand,fol.FOLIfThen):return None
    a=literal(formula.operand.antecedent,fol); b=literal(formula.operand.consequent,fol)
    if a is None or b is None:return None
    return a,b

def predicates(formula, fol):
    out=[]
    def walk(x):
        if isinstance(x,fol.FOLFuncApplication):out.append(x.function);[walk(a) for a in x.args]
        elif isinstance(x,fol.FOLNot):walk(x.operand)
        elif isinstance(x,(fol.FOLForAll,fol.FOLExists)):walk(x.operand)
        elif isinstance(x,(fol.FOLIfThen,fol.FOLIff)):walk(x.antecedent);walk(x.consequent)
        elif isinstance(x,(fol.FOLAnd,fol.FOLOr)):[walk(a) for a in x.operands]
    walk(formula);return out

def clone_formula(x,pmap,mirror_entity,fol):
    if isinstance(x,fol.FOLFuncApplication):return fol.FOLFuncApplication(pmap.get(x.function,x.function),[clone_formula(a,pmap,mirror_entity,fol) for a in x.args])
    if isinstance(x,fol.FOLConstant):return fol.FOLConstant(mirror_entity)
    if isinstance(x,fol.FOLVariable):return fol.FOLVariable(x.variable)
    if isinstance(x,fol.FOLNot):return fol.FOLNot(clone_formula(x.operand,pmap,mirror_entity,fol))
    if isinstance(x,fol.FOLForAll):return fol.FOLForAll(x.variable,clone_formula(x.operand,pmap,mirror_entity,fol))
    if isinstance(x,fol.FOLExists):return fol.FOLExists(x.variable,clone_formula(x.operand,pmap,mirror_entity,fol))
    if isinstance(x,fol.FOLIfThen):return fol.FOLIfThen(clone_formula(x.antecedent,pmap,mirror_entity,fol),clone_formula(x.consequent,pmap,mirror_entity,fol))
    if isinstance(x,fol.FOLIff):return fol.FOLIff(clone_formula(x.antecedent,pmap,mirror_entity,fol),clone_formula(x.consequent,pmap,mirror_entity,fol))
    if isinstance(x,fol.FOLAnd):return fol.FOLAnd([clone_formula(a,pmap,mirror_entity,fol) for a in x.operands])
    if isinstance(x,fol.FOLOr):return fol.FOLOr([clone_formula(a,pmap,mirror_entity,fol) for a in x.operands])
    return x

def opposite(atom,fol):
    return atom.operand if isinstance(atom,fol.FOLNot) else fol.FOLNot(atom)

def adapt(g,concept_names,q,query,formulas,answer,proof):
    import fol
    target=proof[-1].conclusion; target_lit=literal(target,fol)
    if target_lit is None:raise ValueError("non-atomic proof conclusion")
    proof_rules=[s.conclusion for s in proof if isinstance(s.conclusion,fol.FOLForAll)]
    final_rules=[f for f in proof_rules if rule(f,fol) and rule(f,fol)[1]==target_lit]
    if len(final_rules)!=1:raise ValueError(f"expected one final proof rule, got {len(final_rules)}")
    final_rule=final_rules[0]; true_antecedent=rule(final_rule,fol)[0][0]
    opp=(target_lit[0],not target_lit[1])
    dummy=[f for f in formulas if rule(f,fol) and rule(f,fol)[1]==opp and f not in proof_rules]
    if len(dummy)!=1:raise ValueError(f"expected one isolated opposite rule, got {len(dummy)}")
    base=[f for f in formulas if f is not dummy[0] and f!=dummy[0]]
    edges=[]
    for f in base:
        r=rule(f,fol)
        if r:edges.append((r[0][0],r[1][0]))
    adj={}
    for a,b in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
    component={true_antecedent};stack=[true_antecedent]
    while stack:
        u=stack.pop()
        for v in adj.get(u,()):
            if v not in component:component.add(v);stack.append(v)
    used=set(p for f in base for p in predicates(f,fol)); available=[c for c in concept_names if c not in used]
    if len(available)<len(component):raise ValueError("insufficient fresh mirror predicates")
    ordered=sorted(component);chosen=random.sample(available,len(ordered));pmap=dict(zip(ordered,chosen))
    # Proof axiom identifies the queried entity.  Every mirrored fact uses one
    # different entity, preserving fact occurrence without making the decoy
    # reachable for the query entity.
    first=proof[0].conclusion;atom=first.operand if isinstance(first,fol.FOLNot) else first
    selected_entity=atom.args[0].constant
    mirror_entity=next(e for e in g.available_entity_names if e!=selected_entity)
    clone_source=[f for f in base if set(predicates(f,fol)) & component and set(predicates(f,fol)) <= component]
    cloned=[]
    for f in clone_source:
        # Mirror the proof axiom on the queried entity with opposite polarity.
        # The decoy's starting predicate therefore has the same unsigned
        # entity-fact occurrence as the real chain, but the cloned chain is
        # still unreachable because its required signed literal is absent.
        # Other component facts remain on a different entity.
        c=clone_formula(f,pmap,selected_entity if f==first else mirror_entity,fol)
        if f==first:c=opposite(c,fol)
        if f==final_rule:
            assert isinstance(c,fol.FOLForAll) and isinstance(c.operand,fol.FOLIfThen)
            goal=fol.FOLFuncApplication(target_lit[0],[fol.FOLVariable(c.variable)])
            if not target_lit[1]:goal=fol.FOLNot(goal)
            c=fol.FOLForAll(c.variable,fol.FOLIfThen(c.operand.antecedent,goal))
        cloned.append(c)
    transformed=base+cloned
    def render(f):return g.inflect(g.yield_tokens(g.formula_to_clause(f,g.morphology,False)),end_punctuation='.')
    # ``formula_to_clause`` randomly chooses singular/plural wording, so
    # re-rendering the dummy cannot reliably locate its original sentence.
    # The official generator preserves formula/sentence order; remove the
    # corresponding sentence by index instead.
    sentences=[s+'.' for s in q.strip().rstrip('.').split('. ')]
    if len(sentences)!=len(formulas):
        raise ValueError(f"formula/sentence count mismatch: {len(formulas)} != {len(sentences)}")
    dummy_index=next(i for i,f in enumerate(formulas) if f is dummy[0] or f==dummy[0])
    del sentences[dummy_index]
    clone_sentences=[render(f) for f in cloned]
    for sentence,f in zip(clone_sentences,cloned):
        parsed=g.parse_sentence(sentence[:-1],g.morphology,False)
        if parsed!=f:raise ValueError("adapter parser round-trip failed")
    sentences.extend(clone_sentences);random.shuffle(sentences)
    return ' '.join(sentences),query,transformed,answer,proof,{
        'adapter':GENERATOR_VERSION,'mirrored_predicates':len(component),'mirrored_formulas':len(cloned),
        'selected_entity':selected_entity,'mirror_entity':mirror_entity,
        'true_final_antecedent':true_antecedent,'decoy_final_antecedent':pmap[true_antecedent]}
