from .infer import InferModule, ClauseInferModule, ClauseBodyInferModule
from .tensor_encoder import TensorEncoder
from .fol.logic import *
from .fol.data_utils import DataUtils
from .fol.language import DataType

p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.get_clauses(lang)
    bk = du.get_bk(lang)
    atoms = generate_atoms(lang)
    return lang, clauses, bk, atoms


def build_infer_module(clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train)
    return im


def generate_atoms(lang):
    spec_atoms = [false, true]
    atoms = []
    for pred in lang.preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        # args_list = lang.get_args_by_pred(pred)
        args_str_list = []
        # args_mem = []
        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                # if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                # if len(set(args)) == len(args):
                # if not (str(sorted([str(arg) for arg in args])) in args_str_list):
                atoms.append(Atom(pred, args))
                # args_str_list.append(
                #    str(sorted([str(arg) for arg in args])))
                # print('add atom: ', Atom(pred, args))
    return spec_atoms + sorted(atoms)


def build_clause_infer_module(clauses, bk_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    if len(bk_clauses) > 0:
        te_bk = TensorEncoder(lang, atoms, bk_clauses, device=device)
        I_bk = te_bk.encode()
    else:
        te_bk = None
        I_bk = None

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk)
    return im


def build_clause_body_infer_module(clauses, atoms, lang, device, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    # TODO
    im = ClauseBodyInferModule(I, device=device, train=train)
    # im = ClauseInferModule(I, device=device, train=train)
    return im


def get_prednames(clauses):
    prednames = []
    for clause in clauses:
        prednames.append(clause.head.pred.name)
    return prednames


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ['diff_color', 'diff_shape']:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    atoms.append(Atom(pred, args))
    return atoms


def get_index_by_predname(pred_str, atoms):
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
    assert 1, pred_str + ' not found.'


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]


def get_searched_clauses(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.
    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.load_clauses(du.base_path + dataset + '/beam_searched.txt', lang)
    return clauses
