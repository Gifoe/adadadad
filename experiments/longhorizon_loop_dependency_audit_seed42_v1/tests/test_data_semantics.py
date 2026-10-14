import importlib.util,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_data import typed_equal,transition,chains,references,execute

def test_false_is_not_zero():
    assert not typed_equal(False,0) and not typed_equal(True,1)
    assert typed_equal([False],[False]) and not typed_equal([False],[0])
    assert typed_equal(0,0.0)

def test_transition_tables_reversible_and_closed():
    for name in ['ADD','XOR','INV','ROT']:
        assert sorted(transition(i,name) for i in range(8))==list(range(8))

def test_matched_diagnostic_controls_and_rename():
    rows=chains();assert len(rows)==640
    grouped={}
    for row in rows:grouped.setdefault(row['parent_id'],[]).append(row)
    assert len(grouped)==80
    for group in grouped.values():
        assert len(group)==8
        assert len({len(r['prompt']) for r in group})==1 and len({len(r['reference']) for r in group})==1
        for depth in [2,4,8,16]:
            old=next(r for r in group if r['H']==depth and not r['renamed'])
            new=next(r for r in group if r['H']==depth and r['renamed'])
            assert [old['alphabet'].index(x) for x in old['gold_answer']]==[new['alphabet'].index(x) for x in new['gold_answer']]
            assert len(old['dag_edges'])==16-16//depth

def test_structured_reference_is_not_text_similarity():
    found=list(references({'arg0':'$var_2.result$','arg1':['$var_1.result$',2]}))
    assert len(found)==2 and found[0][1:3]==('var_2','result')
    assert not list(references('The word var_2 appears here'))

def test_execution_flag_separate_from_false_value():
    class Methods:
        @staticmethod
        def returns_false():return False
        @staticmethod
        def broken():raise ValueError('failed')
    tools=[{'name':n,'output_parameters':{'result':{}}} for n in ['returns_false','broken']]
    call=lambda name:[{'name':name,'label':'$var_1','arguments':{}}]
    good=execute(call('returns_false'),tools,Methods);bad=execute(call('broken'),tools,Methods)
    assert good['ok'] and good['value'] is False and not bad['ok'] and bad['value'] is None
