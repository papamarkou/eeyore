import torch

import eeyore.stats as st

from .chain_file import ChainFile

class ChainLists:
    def __init__(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        self.reset(keys=keys, vals=vals)

    def reset(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        if vals is None:
            self.vals = {key : [] for key in keys}
        else:
            self.vals = vals

    @classmethod
    def from_chain_list(selfclass, chain_lists, keys=['sample', 'target_val', 'accepted']):
        common_keys = set.intersection(*[set(chain_list.vals.keys()) for chain_list in chain_lists])
        class_keys = set(keys) & common_keys

        vals = {}

        for key in class_keys:
            vals[key] = [chain_list.vals[key] for chain_list in chain_lists]

        return selfclass(keys=class_keys, vals=vals)

    @classmethod
    def from_file(selfclass, paths, keys=['sample', 'target_val', 'accepted'], mode='a', dtype=torch.float64, device='cpu'):
        chain_lists = []

        for path in paths:
            chain_lists.append(ChainFile(keys=keys, path=path, mode=mode).to_chainlist(dtype=dtype, device=device))

        return selfclass.from_chain_list(chain_lists, keys=keys)

    def __repr__(self):
        return f"{len(self)} Markov chains, each containing {self.num_samples()} samples."

    def __len__(self):
        return self.num_chains()

    def num_params(self):
        return len(self.vals['sample'][0][0])

    def num_samples(self):
        return len(self.vals['sample'][0])

    def num_chains(self):
        return len(self.vals['sample'])

    def get_samples(self):
        return torch.stack([self.get_chain(i, key='sample') for i in range(self.num_chains())])

    def get_target_vals(self):
        return torch.stack([self.get_chain(i, key='target_val') for i in range(self.num_chains())])

    def get_grad_vals(self):
        return torch.stack([self.get_chain(i, key='grad_val') for i in range(self.num_chains())])

    def get_chain(self, idx, key='sample'):
        return torch.stack(self.vals[key][idx])

    def mean(self):
        return self.get_samples().mean(1)

    def mean_summary(self, g=lambda x: torch.mean(x, dim=0)):
        return g(self.mean())

    def mc_se(self, mc_cov_mat=None, method='inse', adjust=False):
        return torch.stack([
            st.mc_se(self.get_chain(i, key='sample'), method=method, adjust=adjust, rowvar=False)
            if mc_cov_mat is None
            else st.mc_se_from_cov(mc_cov_mat[i])
            for i in range(self.num_chains())
        ])

    def mc_se_summary(self, g=lambda x: torch.mean(x, dim=0), mc_cov_mat=None, method='inse', adjust=False):
        return g(self.mc_se(mc_cov_mat=mc_cov_mat, method=method, adjust=adjust))

    def mc_cov(self, method='inse', adjust=False):
        return torch.stack([
            st.mc_cov(self.get_chain(i, key='sample'), method=method, adjust=adjust, rowvar=False)
            for i in range(self.num_chains())
        ])

    def mc_cov_summary(self, g=lambda m: torch.mean(m, dim=0), method='inse', adjust=False):
        return g(self.mc_cov(method=method, adjust=adjust))

    def mc_cor(self, mc_cov_mat=None, method='inse', adjust=False):
        return torch.stack([
            st.mc_cor(self.get_chain(i, key='sample'), method=method, adjust=adjust, rowvar=False)
            if mc_cov_mat is None
            else st.cor_from_cov(mc_cov_mat[i])
            for i in range(self.num_chains())
        ])

    def mc_cor_summary(self, g=lambda m: torch.mean(m, dim=0), mc_cov_mat=None, method='inse', adjust=False):
        return g(self.mc_cor(mc_cov_mat=mc_cov_mat, method=method, adjust=adjust))

    def acceptance(self):
        return [sum(self.vals['accepted'][i]) / self.num_samples() for i in range(self.num_chains())]

    def acceptance_summary(self, g=lambda x: sum(x) / len(x)):
        return g(self.acceptance())

    def multi_ess(self, mc_cov_mat=None, method='inse', adjust=False):
        return [
            st.multi_ess(
                self.get_chain(i, key='sample'),
                mc_cov_mat=None if mc_cov_mat is None else mc_cov_mat[i],
                method=method,
                adjust=adjust
            )
            for i in range(self.num_chains())
        ]

    def multi_ess_summary(self, g=lambda x: sum(x) / len(x), mc_cov_mat=None, method='inse', adjust=False):
        return g(self.multi_ess(mc_cov_mat=mc_cov_mat, method=method, adjust=adjust))

    def multi_rhat(self, mc_cov_mat=None, method='inse', adjust=False):
        return st.multi_rhat(self.get_samples(), mc_cov_mat=mc_cov_mat, method=method, adjust=adjust)

    def summary(
        self,
        keys=['multi_ess', 'multi_rhat'],
        g_mean_summary=lambda x: torch.mean(x, dim=0),
        g_mc_se_summary=lambda x: torch.mean(x, dim=0),
        g_acceptance_summary=lambda x: sum(x) / len(x),
        g_multi_ess_summary=lambda x: sum(x) / len(x),
        mc_cov_mat=None,
        method='inse',
        adjust=False):
        summaries = {}

        if any(item in keys for item in ['mc_se', 'multi_ess', 'multi_rhat']):
            if mc_cov_mat is None:
                mc_cov_mat = self.mc_cov(method=method, adjust=adjust)

        for key in keys:
            if key == 'mean':
                summaries[key] = self.mean_summary(g=g_mean_summary)
            elif key == 'mc_se':
                summaries[key] = self.mc_se_summary(g=g_mc_se_summary, mc_cov_mat=mc_cov_mat, method=method, adjust=adjust)
            elif key == 'acceptance':
                summaries[key] = self.acceptance_summary(g=g_acceptance_summary)
            elif key == 'multi_ess':
                summaries[key] = self.multi_ess_summary(
                    g=g_multi_ess_summary, mc_cov_mat=mc_cov_mat, method=method, adjust=adjust
                )
            elif key == 'multi_rhat':
                summaries[key], _, _, _, _ = self.multi_rhat(mc_cov_mat=mc_cov_mat, method=method, adjust=adjust)

        return summaries
