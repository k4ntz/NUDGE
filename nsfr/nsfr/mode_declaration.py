from .fol.language import DataType


class ModeDeclaration(object):
    """from https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html
    p(ModeType, ModeType,...)

    Here are some examples of how they appear in a file:

    :- mode(1,mem(+number,+list)).
    :- mode(1,dec(+integer,-integer)).
    :- mode(1,mult(+integer,+integer,-integer)).
    :- mode(1,plus(+integer,+integer,-integer)).
    :- mode(1,(+integer)=(#integer)).
    :- mode(*,has_car(+train,-car)).
    Each ModeType is either (a) simple; or (b) structured.
    A simple ModeType is one of:
    (a) +T specifying that when a literal with predicate symbol p appears in a
    hypothesised clause, the corresponding argument should be an "input" variable of type T;
    (b) -T specifying that the argument is an "output" variable of type T; or
    (c) #T specifying that it should be a constant of type T.
    All the examples above have simple modetypes.
    A structured ModeType is of the form f(..) where f is a function symbol,
    each argument of which is either a simple or structured ModeType.
    Here is an scripts containing a structured ModeType:


    To make this more clear, here is an scripts for the mode declarations for
    the grandfather task from
     above::- modeh(1, grandfather(+human, +human)).:-
      modeb(*, parent(-human, +human)).:-
       modeb(*, male(+human)).
       The  first  mode  states  that  the  head  of  the  rule
        (and  therefore  the  targetpredicate) will be the atomgrandfather.
         Its parameters have to be of the typehuman.
          The  +  annotation  says  that  the  rule  head  needs  two  variables.
            Thesecond mode declaration states theparentatom and declares again
             that theparameters have to be of type human.
              Here,  the + at the second parametertells, that the system is only allowed to
              introduce the atomparentin the clauseif it already contains a variable of type human.
               The−at the first attribute in-troduces a new variable into the clause.
    The  modes  consist  of  a  recall n that  states  how  many  versions  of  the
    literal are allowed in a rule and an atom with place-markers that state the literal to-gether
    with annotations on input- and output-variables as well as constants (see[Mug95]).
    Args:
        recall (int): The recall number i.e. how many times the declaration can be instanciated
        pred (Predicate): The predicate.
        mode_terms (ModeTerm): Terms for mode declarations.
    """

    def __init__(self, mode_type, recall, pred, mode_terms, ordered=True):
        self.mode_type = mode_type  # head or body
        self.recall = recall
        self.pred = pred
        self.mode_terms = mode_terms
        self.ordered = ordered

    def __str__(self):
        s = 'mode_' + self.mode_type + '('
        for mt in self.mode_terms:
            s += str(mt)
            s += ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class ModeTerm(object):
    """Terms for mode declarations. It has mode (+, -, #) and data types.
    """

    def __init__(self, mode, dtype):
        self.mode = mode
        assert mode in ['+', '-', '#'], "Invalid mode declaration."
        self.dtype = dtype

    def __str__(self):
        return self.mode + self.dtype.name

    def __repr__(self):
        return self.__str__()


def get_mode_declarations_loot(lang):
    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))
    s_type = ModeTerm('#', DataType('type'))
    s_color = ModeTerm('#', DataType('color'))

    modeb_list = [
        ModeDeclaration('body', 2, lang.get_pred_by_name('type'), [p_object, s_type]),
        ModeDeclaration('body', 2, lang.get_pred_by_name('color'), [p_object, s_color]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('close'), [p_object, p_object], ordered=False),
        #ModeDeclaration('body', 1, lang.get_pred_by_name('closeby_vertical'), [p_object, p_object], ordered=False),
        #ModeDeclaration('body', 1, lang.get_pred_by_name('closeby_horizontal'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_top'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('at_bottom'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_left'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_right'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('have_key'), [p_object], ordered=False),
        #ModeDeclaration('body', 1, lang.get_pred_by_name('not_have_key'), [p_image], ordered=False),
    ]
    return modeb_list


def get_mode_declarations_threefish(lang):

    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))
    s_type = ModeTerm('#', DataType('type'))

    modeb_list = [
        ModeDeclaration('body', 2, lang.get_pred_by_name('type'), [p_object, s_type]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('closeby'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_top'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('at_bottom'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_left'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_right'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('is_bigger_than'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('is_smaller_than'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('high_level'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('low_level'), [p_object, p_object]),
    ]
    return modeb_list


def get_mode_declarations_getout(lang):
    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))
    s_type = ModeTerm('#', DataType('type'))

    modeb_list = [
        ModeDeclaration('body', 2, lang.get_pred_by_name('type'), [p_object, s_type]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('closeby'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_left'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('on_right'), [p_object, p_object]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('have_key'), [p_image], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name('not_have_key'), [p_image], ordered=False),
    ]
    return modeb_list


def get_mode_declarations(args, lang):
    if args.m == 'getout':
        return get_mode_declarations_getout(lang)
    if args.m == 'threefish':
        return get_mode_declarations_threefish(lang)
    elif args.m == 'loot':
        return get_mode_declarations_loot(lang)
    else:
        assert False, "Invalid data type."
