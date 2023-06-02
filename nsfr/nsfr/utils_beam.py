import os
from .facts_converter import FactsConverter
from .nsfr_beam import NSFReasoner
from .logic_utils import build_infer_module, build_clause_infer_module, build_clause_body_infer_module
from .valuation_cj import CJValuationModule
from .valuation_bf import BFValuationModule
from .valuation_h import HValuationModule
from nsfr.logic_utils import get_lang


def update_initial_clauses(clauses, obj_num):
    print(len(clauses))
    assert len(clauses) == 1, "Too many initial clauses."
    clause = clauses[0]
    clause.body = clause.body[:obj_num]
    return [clause]


def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):
    VM = CJValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=args.m, infer_step=2, device=device, train=train)
    CIM = build_clause_infer_module(clauses, bk_clauses, atoms, lang,
                                    m=len(clauses), infer_step=2, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC,
                       infer_module=IM, clause_infer_module=CIM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


def get_nsfr_cgen_model(args, lang, clauses, atoms, bk, device, train=False):
    if args.m == 'getout':
        VM = CJValuationModule(lang=lang, device=device)
    elif args.m == 'threefish':
        VM = BFValuationModule(lang=lang, device=device)
    elif args.m == 'loot':
        VM = HValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=args.m, infer_step=2, device=device, train=train)
    CIM = build_clause_body_infer_module(clauses, atoms, lang, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC,
                       infer_module=IM, clause_infer_module=CIM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR
