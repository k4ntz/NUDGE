from nsfr.facts_converter import FactsConverter
from nsfr.nsfr_beam import NSFReasoner
from logic import build_infer_module, build_clause_infer_module, build_clause_body_infer_module
from nsfr.valuation import ValuationModule


def update_initial_clauses(clauses, obj_num):
    print(len(clauses))
    assert len(clauses) == 1, "Too many initial clauses."
    clause = clauses[0]
    clause.body = clause.body[:obj_num]
    return [clause]


def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):
    env_name = args.env
    val_fn_path = f"example/valuation/{env_name}.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=args.m, infer_step=2, device=device, train=train)
    CIM = build_clause_infer_module(clauses, bk_clauses, atoms, lang,
                                    m=len(clauses), infer_step=2, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC,
                       infer_module=IM, clause_infer_module=CIM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


def get_nsfr_cgen_model(args, lang, clauses, atoms, bk, device, train=False):
    env_name = args.env
    val_fn_path = f"example/valuation/{env_name}.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=args.m, infer_step=2, device=device, train=train)
    CIM = build_clause_body_infer_module(clauses, atoms, lang, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC,
                       infer_module=IM, clause_infer_module=CIM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR
