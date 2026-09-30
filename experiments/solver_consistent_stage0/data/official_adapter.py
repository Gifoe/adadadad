"""Extend lexical inputs at an official API boundary; proof and syntax are unchanged."""
import os,sys,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]

def official_generator():
    old=os.getcwd(); vendor=ROOT/'vendor/prontoqa'; sys.path.insert(0,str(vendor)); os.chdir(vendor)
    try: import run_experiment as g
    finally: os.chdir(old)
    concepts=['concept'+str(i) for i in range(2048)]
    for name in concepts:
        if not g.morphology.is_noun(name): g.morphology.add_noun(name,name+'s')
    if not getattr(g,'stage0_lexical_adapter',False):
        original=g.generate_theory
        original_question=g.generate_question
        initialized_ids=set()
        def extended_lexical_inputs(available_concept_names,available_property_families,config):
            # Fresh mutually disjoint property families; repeated adjective strings
            # could create contradictory ontologies, so no old family is duplicated.
            if id(available_property_families) not in initialized_ids:
                extra=[['attribute'+str(4*i+j) for j in range(4)] for i in range(32)]
                available_property_families.extend(extra)
                initialized_ids.add(id(available_property_families))
            # Preserve upstream destructive consumption across main/distractor calls.
            return original(available_concept_names,available_property_families,config)
        def question_with_lexical_scope(*args,**kwargs):
            initialized_ids.clear()
            return original_question(*args,**kwargs)
        g.generate_theory=extended_lexical_inputs
        g.generate_question=question_with_lexical_scope
        g.stage0_lexical_adapter=True
    return g,concepts
