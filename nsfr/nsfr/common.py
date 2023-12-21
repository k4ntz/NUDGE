import os

from nsfr.facts_converter import FactsConverter
from nsfr.utils.logic import get_lang, build_infer_module
from nsfr.nsfr import NSFReasoner
from nsfr.valuation import ValuationModule


def get_nsfr_model(env_name: str, rules: str, device: str, train=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return NSFR
