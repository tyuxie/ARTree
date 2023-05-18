import torch
import numpy as np
import pdb

def decompJC(symm=False):
    pden = np.array([.25, .25, .25, .25])
    rate_matrix_JC = 1.0/3 * np.ones((4,4))
    for i in range(4):
        rate_matrix_JC[i,i] = -1.0
    
    if not symm:
        D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
        U_JC_inv = np.linalg.inv(U_JC)
    else:
        D_JC, W_JC = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_JC), np.diag(np.sqrt(1.0/pden))))
        U_JC = np.dot(np.diag(np.sqrt(1.0/pden)), W_JC)
        U_JC_inv = np.dot(W_JC.T, np.diag(np.sqrt(pden)))
    
    return D_JC, U_JC, U_JC_inv, rate_matrix_JC


def decompHKY(pden, kappa, symm=False):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(pA+pG)*(pC+pT) + 2*kappa*(pA*pG+pC*pT))
    rate_matrix_HKY = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_HKY[i,j] = pden[j]
            if i+j == 1 or i+j == 5:
                rate_matrix_HKY[i,j] *= kappa
    
    for i in range(4):
        rate_matrix_HKY[i,i] = - sum(rate_matrix_HKY[i,])
    
    rate_matrix_HKY = beta * rate_matrix_HKY
    
    if not symm:
        D_HKY, U_HKY = np.linalg.eig(rate_matrix_HKY)
        U_HKY_inv = np.linalg.inv(U_HKY)
    else:
        D_HKY, W_HKY = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_HKY), np.diag(np.sqrt(1.0/pden))))
        U_HKY = np.dot(np.diag(np.sqrt(1.0/pden)), W_HKY)
        U_HKY_inv = np.dot(W_HKY.T, np.diag(np.sqrt(pden)))
       
    return D_HKY, U_HKY, U_HKY_inv, rate_matrix_HKY


def decompGTR(pden, AG, AC, AT, GC, GT, CT, symm=False):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT))
    rate_matrix_GTR = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_GTR[i,j] = pden[j]
                if i+j == 1:
                    rate_matrix_GTR[i,j] *= AG
                if i+j == 2:
                    rate_matrix_GTR[i,j] *= AC
                if i+j == 3 and abs(i-j) > 1:
                    rate_matrix_GTR[i,j] *= AT
                if i+j == 3 and abs(i-j) == 1:
                    rate_matrix_GTR[i,j] *= GC
                if i+j == 4:
                    rate_matrix_GTR[i,j] *= GT
                if i+j == 5:
                    rate_matrix_GTR[i,j] *= CT
    
    for i in range(4):
        rate_matrix_GTR[i,i] = - sum(rate_matrix_GTR[i,])
    
    rate_matrix_GTR = beta * rate_matrix_GTR
    
    if not symm:
        D_GTR, U_GTR = np.linalg.eig(rate_matrix_GTR)
        U_GTR_inv = np.linalg.inv(U_GTR)
    else:
        D_GTR, W_GTR = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_GTR), np.diag(np.sqrt(1.0/pden))))
        U_GTR = np.dot(np.diag(np.sqrt(1.0/pden)), W_GTR)
        U_GTR_inv = np.dot(W_GTR.T, np.diag(np.sqrt(pden)))        
    
    return D_GTR, U_GTR, U_GTR_inv, rate_matrix_GTR

class PHY(object):
    nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.]}
    
    def __init__(self, data, taxa, pden, subModel, scale=0.1, unique_site=True, device=torch.device('cpu')):
        self.ntips = len(data)
        self.nsites = len(data[0])
        self.taxa = taxa
        Qmodel, Qpara = subModel
        if Qmodel == "JC":
            self.D, self.U, self.U_inv, self.rateM = decompJC()  ##We use JC model in VBPI
        if Qmodel == "HKY":
            self.D, self.U, self.U_inv, self.rateM = decompHKY(pden, Qpara)
        if Qmodel == "GTR":
            AG, AC, AT, GC, GT, CT = Qpara
            self.D, self.U, self.U_inv, self.rateM = decompGTR(pden, AG, AC, AT, GC, GT, CT)
        
        self.pden = torch.from_numpy(pden).float().to(device=device)
        self.D = torch.from_numpy(self.D).float().to(device=device)
        self.U = torch.from_numpy(self.U).float().to(device=device)
        self.U_inv = torch.from_numpy(self.U_inv).float().to(device=device)
        
        if unique_site:
            self.L, self.site_counts = map(torch.FloatTensor, self.initialCLV(data, unique_site=True))
            self.L = self.L.to(device=device)
            self.site_counts = self.site_counts.to(device=device)
        else:
            self.L, self.site_counts = torch.FloatTensor(self.initialCLV(data)).to(device=device), 1.0
        self.scale= scale
        self.device = device

    def initialCLV(self, data, unique_site=False):
        if unique_site:
            data_arr = np.array(list(zip(*data)))
            unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
            unique_data = unique_sites.T
            
            return [np.transpose([self.nuc2vec[c] for c in unique_data[i]]) for i in range(self.ntips)], counts
        else:
            return [np.transpose([self.nuc2vec[c] for c in data[i]]) for i in range(self.ntips)]

    def logprior(self, log_branch):
        return -torch.sum(torch.exp(log_branch)/self.scale + np.log(self.scale) - log_branch, -1)
    
    def loglikelihood(self, log_branch, tree):
        branch_D = torch.einsum("i,j->ij", (log_branch.exp(), self.D))
        transition_matrix = torch.matmul(torch.einsum("ij,kj->kij", (self.U, torch.exp(branch_D))), self.U_inv).clamp(0.0)
        scaler_list = []
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.state = self.L[node.name].detach()
            else:
                node.state = 1.0
                for child in node.children:
                    node.state *= transition_matrix[child.name].mm(child.state)
                scaler = torch.sum(node.state, 0)
                node.state /= scaler
                scaler_list.append(scaler)

        scaler_list.append(torch.mm(self.pden.view(-1,4), tree.state).squeeze())
        logll = torch.sum(torch.log(torch.stack(scaler_list)) * self.site_counts)
        return logll
    
    def logp_joint(self, log_branch, tree):
        return self.logprior(log_branch) + self.loglikelihood(log_branch, tree)