from .utils_beam import get_nsfr_cgen_model
from .tensor_encoder import TensorEncoder
from .infer import ClauseBodyInferModule
from .refinement import RefinementGenerator
from tqdm import tqdm
import torch
import numpy as np


class ClauseGenerator(object):
    """
    clause generator by refinement and beam search
    Parameters
    ----------
    ilp_problem : .ilp_problem.ILPProblem
    infer_step : int
        number of steps in forward inference
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, args, NSFR, lang, facts, mode_declarations, device, buffer=None):
        self.args = args
        self.NSFR = NSFR
        self.lang = lang
        self.facts = facts
        self.mode_declarations = mode_declarations
        self.device = device
        self.rgen = RefinementGenerator(lang=lang, mode_declarations=mode_declarations)
        self.buffer = buffer
        self.bce_loss = torch.nn.BCELoss()

    def generate(self, C_0, gen_mode='beam', T_beam=7, N_beam=20, N_max=100):
        """
        call clause generation function with or without beam-searching
        Inputs
        ------
        C_0 : Set[.logic.Clause]
            a set of initial clauses
        gen_mode : string
            a generation mode
            'beam' - with beam-searching
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            set of generated clauses
        """
        if gen_mode == 'beam':
            return self.beam_search(C_0, T_beam=T_beam, N_beam=N_beam, N_max=N_max)
        elif gen_mode == 'naive':
            return self.naive(C_0, N_max=N_max)

    def beam_search_clause(self, clause, T_beam=7, N_beam=20, N_max=100, th=0.98):
        """
        perform beam-searching from a clause
        Inputs
        ------
        clause : Clause
            initial clause
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses
        """
        step = 0
        B = [clause]
        C = set()
        C_dic = {}
        B_ = []

        while step < T_beam:
            # print('Beam step: ', str(step),  'Beam: ', len(B))
            B_new = {}
            refs = []
            for c in B:
                refs_i = self.rgen.refinement_clause(c)
                # remove invalid clauses
                # remove already appeared refs
                refs_i = list(set(refs_i).difference(set(B_)))
                B_.extend(refs_i)
                refs.extend(refs_i)
                C = C.union(set([c]))
                #     print("Added: ", c)

            print('Evaluating ', len(refs), 'generated clauses.')
            loss_list = self.eval_clauses(refs)
            for i, ref in enumerate(refs):
                # check duplication
                # if not self.is_in_beam(B_new, ref, loss_list[i]):
                B_new[ref] = loss_list[i]
                C_dic[ref] = loss_list[i]

                # if len(C) >= N_max:
                #    break
            B_new_sorted = sorted(B_new.items(), key=lambda x: x[1], reverse=True)
            # top N_beam refiements
            B_new_sorted = B_new_sorted[:N_beam]
            # B_new_sorted = [x for x in B_new_sorted if x[1] > th]
            for x in B_new_sorted:
                print(x[1], x[0])
            B = [x[0] for x in B_new_sorted]
            # C = B
            step += 1
            if len(B) == 0:
                break
            # if len(C) >= N_max:
            #    break
        return C

    def is_in_beam(self, B, clause, score):
        """If score is the same, same predicates => duplication
        """
        score = score.detach().cpu().numpy()
        preds = set([clause.head.pred] + [b.pred for b in clause.body])
        y = False
        for ci, score_i in B.items():
            score_i = score_i.detach().cpu().numpy()
            preds_i = set([clause.head.pred] + [b.pred for b in clause.body])
            if preds == preds_i and np.abs(score - score_i) < 1e-2:
                y = True
                # print("duplicated: ", clause, ci)
                break
        return y

    def beam_search(self, C_0, T_beam=7, N_beam=20, N_max=100):
        """
        generate clauses by beam-searching from initial clauses
        Inputs
        ------
        C_0 : Set[.logic.Clause]
            set of initial clauses
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses
        """
        C = set()
        for clause in C_0:
            # searched_clause = self.beam_search_clause(clause, T_beam, N_beam, N_max)
            # C.add(searched_clause[0])
            C = C.union(self.beam_search_clause(
                clause, T_beam, N_beam, N_max))
        C = sorted(list(C))
        print('======= BEAM SEARCHED CLAUSES ======')
        for c in C:
            print(c)
        return C


    def naive(self, C_0, T_beam=7, N_max=100):
        """
        Generate clauses without beam-searching from clauses.
        Inputs
        ------
        C_0 : Set[.logic.Clause]
            set of initial clauses
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses                
        """
        step = 0
        C = set()
        C_next = set(C_0)
        while step < T_beam:
            for c in C_next:
                refs = self.rgen.refinement_clause(c)
                C = C.union(set([c]))
                C_next = C_next.difference(set([c]))
                C_next = C_next.union(set(refs))
                if len(C) >= N_max:
                    break
            if len(C) >= N_max:
                break
        C = sorted(list(C))
        print('======= GENERATED CLAUSES ======')
        for c in C:
            print(c)
        return C

    def eval_clauses(self, clauses):
        C = len(clauses)
        predname = self.get_predname(clauses)

        print("Eval clauses: ", len(clauses))
        NSFR = get_nsfr_cgen_model(self.args, self.lang, clauses, self.NSFR.atoms, self.NSFR.bk, self.device)

        # return sum in terms of buffers
        # take product: action prob of policy * body scores
        # shape: (num_clauses, )
        if self.args.scoring:
            # shape: (num_clauses, num_buffers, num_atoms)
            scores_cba = NSFR.clause_eval(self.buffer.logic_states)
            # shape: (num_clauses, num_buffers)
            body_scores = torch.stack([NSFR.predict(score_i, predname=predname) for score_i in scores_cba])
            # action_probs = self.buffer.action_probs

            # scores = self.scoring(action_probs, body_scores)
            if self.args.m == 'getout':
                action_probs, actions = self.get_action_probs_go(predname)
                # scores = torch.sum(self.buffer.action_buffer * body_scores, dim=1)
                scores = self.scoring(action_probs, body_scores, actions)
            elif self.args.m == 'threefish':
                pass
            elif self.args.m == 'loot':
                action_probs, actions = self.get_action_probs_h(predname)
                scores = self.scoring(action_probs, body_scores, actions)
        else:
            scores = torch.zeros((C,)).to(self.device)
        return scores

    def get_predname(self, clauses):
        predname = clauses[0].head.pred.name
        return predname

    def scoring(self, action_probs, body_scores, actions):
        # action_probs_ = action_probs.unsqueeze(0).expand((body_scores.size(0), -1))
        actions_ = actions.unsqueeze(0).expand((body_scores.size(0), -1))
        # scores = action_probs_ * body_scores
        scores = actions_ * body_scores
        scores = torch.sum(scores, dim=1)

        return scores

    #
    def get_action_probs_go(self, predname):
        # action_probs = torch.stack(self.buffer.action_probs, dim=1).squeeze(0)
        action_probs = self.buffer.action_probs.squeeze(1)
        # action_probs = self.buffer.action_probs.squeeze(1)
        action_list = action_probs.tolist()
        actions = [i.index(max(i)) for i in action_list]
        if 'jump' in predname:
            action_probs = action_probs[:, 2]
            actions = [1 if i == 2 else 0 for i in actions]
        elif 'left' in predname:
            action_probs = action_probs[:, 0]
            actions = [1 if i == 0 else 0 for i in actions]
        elif 'right' in predname:
            action_probs = action_probs[:, 1]
            actions = [1 if i == 1 else 0 for i in actions]
        #
        return action_probs, torch.tensor(actions, device=self.device)

    def get_action_probs_h(self, predname):
        action_probs = self.buffer.action_probs.squeeze(1)
        # action_probs = self.buffer.action_probs.squeeze(1)
        action_list = action_probs.tolist()
        actions = [i.index(max(i)) for i in action_list]
        if 'up' in predname:
            action_probs = action_probs[:, 2]
            actions = [1 if i == 3 else 0 for i in actions]
        elif 'left' in predname:
            action_probs = action_probs[:, 0]
            actions = [1 if i == 0 else 0 for i in actions]
        elif 'right' in predname:
            action_probs = action_probs[:, 1]
            actions = [1 if i == 4 else 0 for i in actions]
        elif 'down' in predname:
            action_probs = action_probs[:, 1]
            actions = [1 if i == 1 else 0 for i in actions]

        return action_probs, torch.tensor(actions, device=self.device)
