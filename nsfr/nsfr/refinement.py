import itertools
from .fol.logic import Atom, Clause, FuncTerm, Var
from .fol.logic_ops import subs


# TODOL refine_from_modeb, generate_by_refinement
class RefinementGenerator(object):
    """
    refinement operations for clause generation
    Parameters
    ----------
    lang : .language.Language
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, lang, mode_declarations):
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.vi = 0  # counter for new variable generation

    def _init_recall_counter_dic(self, mode_declarations):
        dic = {}
        for md in mode_declarations:
            dic[str(md)] = 0
        return dic

    def _check_recall(self, clause, mode_declaration):
        """Return a boolean value that represents the mode declaration can be used or not
        in terms of the recall.
        """
        return clause.count_by_predicate(mode_declaration.pred) < mode_declaration.recall
        # return self.recall_counter_dic[str(mode_declaration)] < mode_declaration.recall

    def _increment_recall(self, mode_declaration):
        self.recall_counter_dic[str(mode_declaration)] += 1

    def get_max_obj_id(self, clause):
        object_vars = clause.all_vars_by_dtype('object')
        object_ids = [int(x.name.split('O')[-1]) for x in object_vars]
        if len(object_ids) == 0:
            return 0
        else:
            return max(object_ids)

    # def __generate_new_variable(self, n):

    # We assume that we have only object variables as new variables
    # O1, O2, ....
    # new_var = Var('O' + str(self.object_counter))
    # new_var = Var("__Y" + str(self.vi) + "__")
    # self.vi += 1
    # return new_var
    # new_var = Var('O' + str(n+1))
    # self.object_counter += 1
    # return new_var

    def generate_new_variable(self, clause):
        obj_id = self.get_max_obj_id(clause)
        return Var('O' + str(obj_id + 1))

    def refine_from_modeb(self, clause, modeb):
        """Generate clauses by adding atoms to body using mode declaration.

        Args:
              clause (Clause): A clause.
              modeb (ModeDeclaration): A mode declaration for body.
        """
        # list(list(Term))

        if not self._check_recall(clause, modeb):
            # the input modeb has been used as many as its recall (maximum number to be called) already
            return []

        terms_list = self.generate_term_combinations(clause, modeb)

        C_refined = []
        for terms in terms_list:
            if len(terms) == len(list(set(terms))):
                # terms: (O0, X)
                if not modeb.ordered:
                    terms = sorted(terms)
                new_atom = Atom(modeb.pred, terms)
                if not new_atom in clause.body:
                    new_clause = Clause(clause.head, clause.body + [new_atom])
                    C_refined.append(new_clause)
        # self._increment_recall(modeb)
        return list(set(C_refined))

    def generate_term_combinations(self, clause, modeb):
        """Generate possible term list for new body atom.
        Enumerate possible assignments for each place in the mode predicate,
        generate all possible assignments by enumerating the combinations.

        Args:
            modeb (ModeDeclaration): A mode declaration for body.
        """
        assignments_list = []
        for mt in modeb.mode_terms:
            if mt.mode == '+':
                # var_candidates = clause.var_all()
                assignments = clause.all_vars_by_dtype(mt.dtype)
            elif mt.mode == '-':
                # get new variable
                # How to think as candidates? maybe [O3] etc.
                # we get only object variable e.g. O3
                # new_var = self.generate_new_variable()
                assignments = [self.generate_new_variable(clause)]
            elif mt.mode == '#':
                # consts = self.lang.get_by_dtype(mt.mode.dtype)
                assignments = self.lang.get_by_dtype(mt.dtype)
            assignments_list.append(assignments)
        # generate all combinations by cartesian product
        # e.g. [[O2], [red,blue,yellow]]
        # -> [[O2,red],[O2,blue],[O2,yellow]]
        ##print(assignments_list)
        ##print(list(itertools.product(*assignments_list)))
        ##print(clause, modeb, assignments_list)
        # print(clause, modeb)
        # print(assignments_list)
        if modeb.ordered:
            return itertools.product(*assignments_list)
        else:
            return itertools.combinations(assignments_list[0], modeb.pred.arity)

    def refinement_clause(self, clause):
        C_refined = []
        for modeb in self.mode_declarations:
            new_clauses = self.refine_from_modeb(clause, modeb)
            # new_clauses = [c for c in new_clauses if self._is_valid(c)]
            C_refined.extend(new_clauses)
            ##print(C_refined)
        return list(set(C_refined))

    def refinement(self, clauses):
        """Perform refinement for given set of clauses.

        Args:
            clauses (list(Clauses)): A set of clauses.
        Returns:
            list(Clauses): A set of refined clauses using modeb declarations.
        """
        result = []
        for clause in clauses:
            C_refined = self.refinement_clause(clause)
            for c in C_refined:
                if not (c in result):
                    result.append(c)
        return result

    def __refinement_clauses(self, C):
        """
        apply refinement operations to each element in given set of clauses
        Inputs
        ------
        C : List[.logic.Clause]
            set of clauses
        Returns
        -------
        C_refined : List[.logic.Clause]
            refined clauses
        """
        C_refined = []
        for clause in C:
            C_refined.extend(self.refinement(clause))
        return list(set(C_refined))

    def ___refinement(self, clause):
        """
        refinement operator that consist of 4 types of refinement
        Inputs
        ______
        clause : .logic.Clause
            input clause to be refined
        Returns
        -------
        refined_clauses : List[.logic.Clause]
            refined clauses
        """
        # refs = list(set(self.add_atom(clause) + self.apply_func(clause) +
        #                self.subs_var(clause) + self.subs_const(clause)))
        # refs = list(set(self.add_atom(clause)))
        refs = list(set(self.add_attribute_atom(clause) + self.add_relation_atom(clause)))
        result = []
        for ref in refs:
            if not '' in [str(arg) for arg in ref.head.terms]:
                result.append(ref)
        return result

    def add_atom(self, clause):
        """
        add p(x_1, ..., x_n) to the body
        """
        # Check body length
        if (len(clause.body) >= self.max_body_len) or (len(clause.all_consts()) >= 1):
            return []

        refined_clauses = []
        for p in self.lang.preds:
            var_candidates = clause.all_vars()
            # Select X_1, ..., X_n for new atom p(X_1, ..., X_n)
            # 1. Selection 2. Ordering
            for vs in itertools.permutations(var_candidates, p.arity):
                new_atom = Atom(p, vs)
                head = clause.head
                if new_atom != head and not (new_atom in clause.body):
                    new_body = clause.body + [new_atom]
                    new_clause = Clause(head, new_body)
                    refined_clauses.append(new_clause)
        return refined_clauses

    def add_attribute_atom(self, clause):
        refined_clauses = []
        for p in self.mode_declarations.get_attribute_preds:
            var_candidates = clause.all_vars()
            # Select X_1, ..., X_n for new atom p(X_1, ..., X_n)
            # 1. Selection 2. Ordering
            assert len(p.dtypes) == 2, "Invalid arity in refinement for attribute atoms, arity: " + str(len(p.dtypes))
            attr_dtype = p.dtypes[-1]
            for v in var_candidates:
                consts = self.lang.get_by_dtype(attr_dtype)
                for c in consts:
                    # add attribute atom to body
                    new_atom = Atom(p, [v, c])
                    new_body = clause.body + [new_atom]
                    new_clause = Clause(clause.head, new_body)
                    refined_clauses.append(new_clause)
        return refined_clauses

    def add_relation_atom(self, clause):
        refined_clauses = []
        for p in self.mode_manager.get_relational_preds():
            var_candidates = clause.all_vars()
            # Select X_1, ..., X_n for new atom p(X_1, ..., X_n)
            # 1. Selection 2. Ordering
            for vs in itertools.permutations(var_candidates, p.arity):
                new_atom = Atom(p, vs)
                head = clause.head
                if new_atom != head and not (new_atom in clause.body):
                    new_body = clause.body + [new_atom]
                    new_clause = Clause(head, new_body)
                    refined_clauses.append(new_clause)
        return refined_clauses

    def _add_atom(self, clause):
        """
        add p(x_1, ..., x_n) to the body
        """
        # Check body length
        if (len(clause.body) >= self.max_body_len) or (len(clause.all_consts()) >= 1):
            return []

        refined_clauses = []
        for p in self.lang.preds:
            var_candidates = clause.all_vars()
            # Select X_1, ..., X_n for new atom p(X_1, ..., X_n)
            # 1. Selection 2. Ordering
            for vs in itertools.permutations(var_candidates, p.arity):
                new_atom = Atom(p, vs)
                head = clause.head
                if new_atom != head and not (new_atom in clause.body):
                    new_body = clause.body + [new_atom]
                    new_clause = Clause(head, new_body)
                    refined_clauses.append(new_clause)
        return refined_clauses

    def apply_func(self, clause):
        """
        z/f(x_1, ..., x_n) for every variable in C and every n-ary function symbol f in the language
        """
        refined_clauses = []
        if (len(clause.body) >= self.max_body_len) or (len(clause.all_consts()) >= 1):
            return []

        funcs = clause.all_funcs()
        for z in clause.head.all_vars():
            # for z in clause.all_vars():
            for f in self.lang.funcs:
                # if len(funcs) >= 1 and not(f in funcs):
                #    continue

                new_vars = [self.lang.var_gen.generate()
                            for v in range(f.arity)]
                func_term = FuncTerm(f, new_vars)
                # TODO: check variable z's depth
                result = subs(clause, z, func_term)
                if result.max_depth() <= self.max_depth:
                    result.rename_vars()
                    refined_clauses.append(result)
        return refined_clauses

    def subs_var(self, clause):
        """
        z/x for every distinct variables x and z in C
        """
        refined_clauses = []
        # to HEAD
        all_vars = clause.head.all_vars()
        combs = itertools.combinations(all_vars, 2)
        for u, v in combs:
            result = subs(clause, u, v)
            result.rename_vars()
            refined_clauses.append(result)
        return refined_clauses

    def subs_const(self, clause):
        """
        z/a for every variable z in C and every constant a in the language
        """
        if (len(clause.body) >= self.max_body_len) or (clause.max_depth() >= 1):
            return []

        refined_clauses = []
        all_vars = clause.head.all_vars()
        consts = self.lang.subs_consts
        for v, c in itertools.product(all_vars, consts):
            result = subs(clause, v, c)
            result.rename_vars()
            refined_clauses.append(result)
        return refined_clauses
