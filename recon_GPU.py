import sys
import os

os.environ['TOOLBOX_PATH'] = '/data/ryy/homebackup/dl_project/bart'
sys.path.append('/data/ryy/homebackup/dl_project/bart' + '/python')
from bart import bart
import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import h5py
import scipy.interpolate as si
from utils import *
import torch.nn.functional as F


class Eop():
    def __init__(self):
        super(Eop, self).__init__()

    def mtimes(self, b, inv, sens, us_mask):
        if inv:
            # b: nv,nt,nc,x,y,z
            x = torch.sum(k2i_torch(b * us_mask, ax=[-3, -2, -1]) * torch.conj(sens), dim=2)
        else:
            b = b.unsqueeze(2) * sens
            x = i2k_torch(b, ax=[-3, -2, -1]) * us_mask
        return x


def SoftThres(X, reg):
    X = torch.sgn(X) * (torch.abs(X) - reg) * ((torch.abs(X) - reg) > 0)
    return X


def SVT(X, reg):
    Np, Nt, FE, PE, SPE=X.shape
    reg *= (np.sqrt(np.prod(X.shape[-3:])) + 1)
    U, S, Vh = torch.linalg.svd(X.view(Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    X = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Np, Nt, FE, PE, SPE)
    return X, torch.sum(S_new)


def SVT_LLR(X, reg, blk):
    def GETWIDTH(M, N, B):
        temp = (np.sqrt(M) + np.sqrt(N))
        if M > N:
            return temp + np.sqrt(np.log2(B * N))
        else:
            return temp + np.sqrt(np.log2(B * M))

    Np, Nt, FE, PE, SPE = X.shape
    stepx = np.ceil(FE / blk)
    stepy = np.ceil(PE / blk)
    stepz = np.ceil(SPE / blk)
    padx = (stepx * blk).astype('uint16')
    pady = (stepy * blk).astype('uint16')
    padz = (stepz * blk).astype('uint16')
    rrx = torch.randperm(blk)[0]
    rry = torch.randperm(blk)[0]
    rrz = torch.randperm(blk)[0]
    X = F.pad(X, (0, padx - FE, 0, pady - PE, 0, padz - SPE))
    X = torch.roll(X, (rrz, rry, rrx), (-1, -2, -3))
    FEp, PEp, SPEp = X.shape[-3:]
    patches = X.unfold(2, blk, blk).unfold(3, blk, blk).unfold(4, blk, blk)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(Np, Nt, -1, blk, blk, blk).permute((2, 0, 1, 3, 4, 5))
    Nb = patches.shape[0]
    M = blk ** 3
    N = blk ** 3 / M
    B = FEp * PEp * SPEp / blk ** 3
    RF = GETWIDTH(M, N, B)
    reg *= RF
    U, S, Vh = torch.linalg.svd(patches.view(Nb, Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    patches = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Nb, Np, Nt, blk, blk, blk)
    patches = patches.permute((1, 2, 0, 3, 4, 5))
    patches_orig = patches.view(unfold_shape)
    patches_orig = patches_orig.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    patches_orig = patches_orig.view(Np, Nt, FEp, PEp, SPEp)
    patches_orig = torch.roll(patches_orig, (-rrz, -rry, -rrx), (-1, -2, -3))
    X = patches_orig[..., :FE, :PE, :SPE]
    return X, torch.sum(S_new)


def low_rank_ISTA(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    lossp = 0
    loss_list = np.zeros((it))
    for i in range(it):
        X, lnu = SVT(X, reg)
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        X = X - A.mtimes(axb, 1, csm, us_mask)
        l2 = torch.sum(torch.abs(axb) ** 2)
        lnu = torch.abs(lnu)
        loss = l2 + lnu
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, "%")
        lossp = loss
        loss_list[i] = loss
    return X, loss_list


def low_rank_BART(A, Kv, csm, reg, it):
    Kv = Kv.cpu().numpy()
    csm = csm.cpu().numpy()
    Np, Nt, Nc, FE, PE, SPE = Kv.shape
    Kv = Kv.transpose((3, 4, 5, 2, 1, 0))
    csm = csm.transpose((1, 2, 3, 0))
    bart_string = 'pics -u1 -w 1 -H -d5 -i %d -R L:7:0:%.3e -g' % (it, reg)
    X = bart(1, bart_string, Kv[:, :, :, :, None, None, None, None, None, None, :, :], csm)
    X = np.transpose(np.squeeze(X), [4, 3, 0, 1, 2])
    return X


def low_rank_FISTA(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    lossp = 0
    tp = 1
    Y = X.clone()
    Xp = X.clone()
    loss_list = np.zeros((it))

    def PL(X, Kv, csm, us_mask, reg):
        X, lnu = SVT(X, reg)
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        X = X - A.mtimes(axb, 1, csm, us_mask)
        l2 = torch.sum(torch.abs(axb) ** 2)
        lnu = torch.abs(lnu)
        return X, l2, lnu

    for i in range(it):
        X, l2, lnu = PL(Y, Kv, csm, us_mask, reg)
        t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
        Y = X + (tp - 1) / t * (X - Xp)
        loss = l2 + lnu
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, "%")
        lossp = loss
        tp = t
        Xp = X
        loss_list[i] = loss
    return X, loss_list


def low_rank_FISTA_prox(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    lossp = 0
    tp = 1
    Y = X.clone()
    Xp = X.clone()
    loss_list = np.zeros((it))

    def PL(X, Kv, csm, us_mask, reg):
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        X = X - A.mtimes(axb, 1, csm, us_mask)
        X, lnu = SVT(X, reg)
        l2 = torch.sum(torch.abs(axb) ** 2)
        lnu = torch.abs(lnu)
        return X, l2, lnu

    for i in range(it):
        X, l2, lnu = PL(Y, Kv, csm, us_mask, reg)
        t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
        Y = X + (tp - 1) / t * (X - Xp)
        loss = l2 + lnu
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, "%")
        lossp = loss
        tp = t
        Xp = X
        loss_list[i] = loss
    return X, loss_list


def low_rank_POGM(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    lossp = 0
    t = 1
    Xp, Wp, W, Zp, Z = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
    thetap = 1
    gamma = 1
    loss_list = np.zeros((it))
    for i in range(it):
        if i == it - 1:
            theta = 1 / 2 * (1 + np.sqrt(8 * thetap ** 2 + 1))
        else:
            theta = 1 / 2 * (1 + np.sqrt(4 * thetap ** 2 + 1))
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        W = Xp - t * A.mtimes(axb, 1, csm, us_mask)
        Z = W + (thetap - 1) / theta * (W - Wp) + thetap / theta * (W - Xp) + t / gamma * (thetap - 1) / theta * (
                Zp - Xp)
        gamma = t * (2 * thetap + theta - 1) / theta
        X, lnu = SVT(Z, reg * gamma)
        Xp = X
        Wp = W
        Zp = Z
        thetap = theta
        l2 = torch.sum(torch.abs(axb) ** 2)
        loss = l2 + lnu
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, l2, lnu)
        lossp = loss
        loss_list[i] = loss
    return X, loss_list


def LLR_POGM(A, Kv, csm, us_mask, reg, it, blk):
    X = A.mtimes(Kv, 1, csm, us_mask)
    lossp = 0
    t = 1
    Xp, Wp, W, Zp, Z = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
    thetap = 1
    gamma = 1
    loss_list = np.zeros((it))
    for i in range(it):
        if i == it - 1:
            theta = 1 / 2 * (1 + np.sqrt(8 * thetap ** 2 + 1))
        else:
            theta = 1 / 2 * (1 + np.sqrt(4 * thetap ** 2 + 1))
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        W = Xp - t * A.mtimes(axb, 1, csm, us_mask)
        Z = W + (thetap - 1) / theta * (W - Wp) + thetap / theta * (W - Xp) + t / gamma * (thetap - 1) / theta * (
                Zp - Xp)
        gamma = t * (2 * thetap + theta - 1) / theta
        X, lnu = SVT_LLR(Z, reg * gamma, blk)
        Xp = X
        Wp = W
        Zp = Z
        thetap = theta
        l2 = torch.sum(torch.abs(axb) ** 2)
        loss = l2 + lnu
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, l2, lnu)
        lossp = loss
        loss_list[i] = loss
    return X, loss_list


def MSLLR_ADMM(A, Kv, csm, us_mask, reg, it, blk):
    def GETWIDTH(M, N, B):
        temp = (np.sqrt(M) + np.sqrt(N))
        if M > N:
            return temp + np.sqrt(np.log2(B * N))
        else:
            return temp + np.sqrt(np.log2(B * M))

    X = A.mtimes(Kv, 1, csm, us_mask)
    Np, Nt, FE, PE, SPE = X.shape
    L = np.ceil(np.max(np.log2(X.shape[-3:]))).astype('uint16')
    skip = 2
    blk = []
    for i in range(0, L, skip):
        blk.append([Nt, np.min([2 ** i, FE]), np.min([2 ** i, PE]), np.min([2 ** i, SPE])])
    blk = np.array(blk)
    levels = blk.shape[0]
    ms = np.prod(blk[:, 1:], axis=1)
    ns = blk[:, 0]
    bs = np.repeat(np.prod(X.shape[-3:]), levels) / ms
    X_it = torch.zeros((levels, Np, Nt, FE, PE, SPE)).to(torch.complex64).cuda()
    Z_it = X_it.clone()
    U_it = X_it.clone()
    k = 0
    K = 1
    rho = 0.5
    rho_k = rho
    print("BLOCK SIZE:", blk)
    for i in range(it):
        for j in range(levels):
            axb = Kv - A.mtimes(Z_it[j] - U_it[j], 0, csm, us_mask)
            X_it[j] = A.mtimes(axb, 1, csm, us_mask) + Z_it[j] - U_it[j]
        for l in range(levels):
            print("IT:", i, 'LEVEL:', l)
            XU = X_it[l] + U_it[l]
            stepx = np.ceil(FE / blk[l, 1])
            stepy = np.ceil(PE / blk[l, 2])
            stepz = np.ceil(SPE / blk[l, 3])
            padx = (stepx * blk[l, 1]).astype('uint16')
            pady = (stepy * blk[l, 2]).astype('uint16')
            padz = (stepz * blk[l, 3]).astype('uint16')
            XU = F.pad(XU, (0, padz - SPE, 0, pady - PE, 0, padx - FE))
            rrx = torch.randperm(blk[l, 1])[0]
            rry = torch.randperm(blk[l, 2])[0]
            rrz = torch.randperm(blk[l, 3])[0]
            XU = torch.roll(XU, (rrz, rry, rrx), (-1, -2, -3))
            FEp, PEp, SPEp = XU.shape[-3:]
            patches = XU.unfold(2, blk[l, 1], blk[l, 1]).unfold(3, blk[l, 2], blk[l, 2]).unfold(4, blk[l, 3], blk[l, 3])
            unfold_shape = patches.size()
            patches = patches.contiguous().view(Np, Nt, -1, blk[l, 1], blk[l, 2], blk[l, 3]).permute((2, 0, 1, 3, 4, 5))
            Nb = patches.shape[0]
            U, S, Vh = torch.linalg.svd(patches.view(Nb, Np * Nt, -1), full_matrices=False)
            S_new = SoftThres(S, reg * GETWIDTH(ms[l], ns[l], bs[l]))
            S_new = torch.diag_embed(S_new).to(torch.complex64)
            patches = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Nb, Np, Nt, blk[l, 1], blk[l, 2],
                                                                                  blk[l, 3])
            patches = patches.permute((1, 2, 0, 3, 4, 5))
            patches_orig = patches.view(unfold_shape)
            patches_orig = patches_orig.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
            patches_orig = patches_orig.view(Np, Nt, FEp, PEp, SPEp)
            patches_orig = torch.roll(patches_orig, (-rrz, -rry, -rrx), (-1, -2, -3))
            Z_it[l] = patches_orig[..., :FE, :PE, :SPE]
        U_it = U_it - Z_it + X_it
        k = k + 1
        if (k == K):
            rho_k *= 2
            U_it /= 2
            K = K * 2
            k = 0
    return X_it


def LplusS(A, Kv, csm, us_mask, regl, regs, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    S, Lp, Sp = torch.zeros_like(X), torch.zeros_like(X), torch.zeros_like(X)
    lossp = 0
    loss_list = np.zeros((it))

    def Sparse(S, reg, ax=[0, 1]):
        temp = SoftThres(i2k_torch(S, ax=ax), reg)
        return k2i_torch(temp, ax=ax), torch.sum(torch.abs(temp))

    for i in range(it):
        L, lnu = SVT(X - Sp, regl)
        S, ls = Sparse(X - Lp, regs, ax=[0, 1])
        axb = A.mtimes(L + S, 0, csm, us_mask) - Kv
        X = L + S - A.mtimes(axb, 1, csm, us_mask)
        l2 = torch.sum(torch.abs(axb) ** 2)
        lnu = torch.abs(lnu)
        ls = torch.abs(ls)
        loss = l2 + lnu + ls
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, l2, lnu, ls)
        lossp = loss
        loss_list[i] = loss
        Lp = L
        Sp = S
    return X, loss_list, L, S


def LplusS_POGM(A, Kv, csm, us_mask, regl, regs, it):
    regl *= (np.sqrt(np.prod(Kv.shape[-3:])) + 1)
    M = A.mtimes(Kv, 1, csm, us_mask)
    X = torch.concat([M.clone().unsqueeze(0), torch.zeros_like(M).unsqueeze(0)], dim=0)
    X_, Xh, Xhp = X.clone(), X.clone(), X.clone()
    thetap, kesp = 1, 1
    lossp = 0
    loss_list = np.zeros((it))
    t = 0.5

    def Sparse(S, reg, ax=[0, 1]):
        temp = SoftThres(i2k_torch(S, ax=ax), reg)
        return k2i_torch(temp, ax=ax), torch.sum(torch.abs(temp))

    for i in range(it):
        Xh[0] = M - X[1]
        Xh[1] = M - X[0]
        if i == it - 1:
            theta = (1 + np.sqrt(1 + 8 * thetap ** 2)) / 2
        else:
            theta = (1 + np.sqrt(1 + 4 * thetap ** 2)) / 2
        X_ = Xh + (thetap - 1) / theta * (Xh - Xhp) + thetap / theta * (Xh - X) \
             + (thetap - 1) / theta / kesp * t * (X_ - X)
        kes = t * (1 + (thetap - 1) / theta + thetap / theta)
        X[0], lnu = SVT(X_[0], regl)
        X[1], ls = Sparse(X_[1], regs, ax=[0, 1])
        axb = A.mtimes(X[0] + X[1], 0, csm, us_mask) - Kv
        M = X[0] + X[1] - A.mtimes(axb, 1, csm, us_mask) * t
        l2 = torch.sum(torch.abs(axb) ** 2)
        lnu = torch.abs(lnu)
        ls = torch.abs(ls)
        loss = l2 + lnu + ls
        print("iter----", i, "loss:", (loss - lossp) / loss * 100, l2, lnu, ls)
        lossp = loss
        loss_list[i] = loss
        kesp = kes
        thetap = theta
        Xhp = Xh
    return M, loss_list, X[0], X[1]


if __name__ == '__main__':
    import matplotlib
    import time

    hv = 50
    lv = 150
    img_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    flow_norm = matplotlib.colors.Normalize(vmin=-hv, vmax=hv)
    flow_norm_lv = matplotlib.colors.Normalize(vmin=-lv, vmax=lv)
    flow_norm_dv = matplotlib.colors.Normalize(vmin=-hv, vmax=hv)
    for path in [
        #  '/data1/ryy/5dflow/data/vol/hhz/10_L0_P14/',
        #  '/data1/ryy/5dflow/data/vol/xzc/20240224_222558_xzc_5815/4_R20_A4/',
        #  '/data1/ryy/5dflow/data/vol/lzf/4_R24_P12/',
        #  '/data1/ryy/5dflow/data/vol/lr/20240217_153641_LR_5780/5_R3_A3/',
        #  '/data1/ryy/5dflow/data/vol/hht/4_L13_A3/',
        #  '/data1/ryy/5dflow/data/vol/ryy/4_R8_A2/',
        '/data1/ryy/5dflow/data/vol/zr/4_R4_P2/',
        # '/data1/ryy/5dflow/data/vol/jth/4_L22_A2/',
    ]:
        name = '/'
        Np = 4
        Nt = 10
        VPS = 10
        ratio = 0.6
        cardiac_ecg = 0
        mmwr_sg = 0
        FE_cut = 1
        pad2 = 320
        if ratio != 1:
            type = str(Nt) + "_" + str(ratio)
        else:
            type = str(Nt)
        csm = np.load(path + name + '/sens_final' + str(type) + '.npy')
        kspc = np.load(path + name + '/kspc_cc_final10_0.6.npy').astype('complex64')
        FE, PE, SPE, Nc, _, _, _, _, Nv, _, Nt, Np = kspc.shape
        kspc = kspc[:, :, :, :, 0, 0, 0, 0, :, 0, :, :]
        K = np.transpose(kspc, [4, 6, 5, 3, 0, 1, 2])
        csm = np.transpose(csm, [3, 0, 1, 2])
        A = Eop()
        reg = 0.02
        regS = 0.06
        blk = 8
        # reg * regFactor * (np.sqrt(np.prod(X.shape[-3:])) + 1)
        it = 15
        img = np.zeros_like(K)[:, :, :, 0]

        csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to('cuda')
        for mode in ["POGM"]:
            # mode = 'FISTA'
            loss = np.zeros((Nv, it))
            print("START RECON")
            if 'LS' in mode:
                L_list = np.zeros_like(img)
                S_list = np.zeros_like(img)
            for v in range(Nv):
                Kv = torch.as_tensor(np.ascontiguousarray(K[v])).to(torch.complex64).to('cuda')
                us_mask = (torch.abs(Kv[:, :, 0:1, 0:1]) > 0).to(torch.float32).to('cuda')
                print(us_mask.shape)
                print("US rate:", 1 / torch.mean(us_mask))
                print(Kv.shape)
                print('Sucessfuly loaded data, start recon')
                # regFactor = 0
                sos = torch.sqrt(torch.sum(torch.abs(csm) ** 2, 0)) + 1e-6
                rcomb = torch.sum(k2i_torch(Kv, ax=[-3, -2, -1]) * torch.conj(csm), 2) / sos
                regFactor = torch.max(torch.abs(rcomb))
                print('scaling Factor: ', regFactor)
                Kv /= regFactor
                del rcomb
                del sos
                # X = low_rank_BART(Kv, csm, reg * regFactor, it)
                st = time.time()

                if mode == 'FISTA':
                    X, loss[v] = low_rank_FISTA(A, Kv, csm, us_mask, reg, it)
                    img[v] = X.cpu().numpy()
                elif mode == 'FISTA_prox':
                    X, loss[v] = low_rank_FISTA_prox(A, Kv, csm, us_mask, reg, it)
                    img[v] = X.cpu().numpy()
                elif mode == 'ISTA':
                    X, loss[v] = low_rank_ISTA(A, Kv, csm, us_mask, reg, it)
                    img[v] = X.cpu().numpy()
                elif mode == 'POGM':
                    X, loss[v] = low_rank_POGM(A, Kv, csm, us_mask, reg, it)
                    img[v] = X.cpu().numpy()
                elif mode == 'BART':
                    X = low_rank_BART(A, Kv, csm, reg, it)
                    img[v] = X
                elif mode == "LS":
                    X, loss[v], Lv, Sv = LplusS(A, Kv, csm, us_mask, reg,
                                                regS, it)
                    img[v] = X.cpu().numpy()
                    L_list[v] = Lv.cpu().numpy()
                    S_list[v] = Sv.cpu().numpy()
                elif mode == 'LS_POGM':
                    X, loss[v], Lv, Sv = LplusS_POGM(A, Kv, csm, us_mask, reg,
                                                     regS, it)
                    img[v] = X.cpu().numpy()
                    L_list[v] = Lv.cpu().numpy()
                    S_list[v] = Sv.cpu().numpy()
                elif mode == 'LLR_POGM':
                    X, loss[v] = LLR_POGM(A, Kv, csm, us_mask, reg, it, blk)
                    img[v] = X.cpu().numpy()
                elif mode == 'MSLLR_ADMM':
                    X = MSLLR_ADMM(A, Kv, csm, us_mask, reg, it, blk)
                    X = torch.sum(X, dim=0)
                print("TIME:", time.time() - st)

            np.save(path + name + '/recon_test_' + mode + '.npy', img)
            flow = np.angle(img[1:] * np.conj(img[0:1]))
            mag = np.mean(np.abs(img), axis=0)
            mag /= np.max(mag)
            if "LS" in mode:
                L_list = np.mean(np.abs(L_list), axis=0)
                L_list /= np.max(L_list)
                S_list = np.mean(np.abs(S_list), axis=0)
                S_list /= np.max(S_list)
            SC = SPE // 2 - 1
            P = 0
            if "LS" in mode:
                row = 3
            else:
                row = 2
            col = 4
            for T in range(5):
                plt.figure(figsize=(col * 10, row * 10))
                plt.title(mode, fontsize=15)
                plt.subplot(row, 4, 1)
                plt.imshow(mag[P, T, :, :, SC], cmap='gray', norm=img_norm)
                plt.subplot(row, 4, 2)
                plt.imshow(flow[0, P, T, :, :, SC] / np.pi * lv, cmap='gray', norm=flow_norm_lv)
                plt.subplot(row, 4, 3)
                plt.imshow(flow[1, P, T, :, :, SC] / np.pi * lv, cmap='gray', norm=flow_norm_lv)
                plt.subplot(row, 4, 4)
                plt.imshow(flow[2, P, T, :, :, SC] / np.pi * lv, cmap='gray', norm=flow_norm_lv)
                plt.subplot(row, 4, 5)
                plt.imshow(mag[P, T, :, :, SC], cmap='gray', norm=img_norm)
                plt.subplot(row, 4, 6)
                plt.imshow(flow[3, P, T, :, :, SC] / np.pi * hv, cmap='gray', norm=flow_norm)
                plt.subplot(row, 4, 7)
                plt.imshow(flow[4, P, T, :, :, SC] / np.pi * hv, cmap='gray', norm=flow_norm)
                plt.subplot(row, 4, 8)
                plt.imshow(flow[5, P, T, :, :, SC] / np.pi * hv, cmap='gray', norm=flow_norm)
                if 'LS' in mode:
                    plt.subplot(row, 4, 9)
                    plt.imshow(L_list[P, T, :, :, SC], cmap='gray', norm=img_norm)
                    plt.subplot(row, 4, 10)
                    plt.imshow(np.transpose(L_list[P, :, :, 105, SC], (0, 1)), cmap='gray', norm=img_norm)
                    plt.subplot(row, 4, 11)
                    plt.imshow(S_list[P, T, :, :, SC], cmap='gray')
                    plt.colorbar()
                    plt.subplot(row, 4, 12)
                    plt.imshow(np.transpose(S_list[P, :, :, 105, SC], (0, 1)), cmap='gray')
                    plt.colorbar()
                plt.tight_layout()
                plt.show()
            # plt.plot(np.mean(loss, axis=0), label=mode)
            # plt.legend()
        # plt.show()
