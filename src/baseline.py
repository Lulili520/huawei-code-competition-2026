"""NVFP4 → HiF4 calibration-aware converter.

  r = hif4_calibration_and_quantize_weight(w_q, w_s, [(a_q,a_s),...])
  W = hif4_decode(*r["weight_params"])
  A = hif4_decode(*hif4_quantize_activation(a_q, a_s, r["activation_state"]))

  st = hif4_calibration_attention([{"q":..,"k":..,"v":..},...], qH, kvH, D)
  q, k, v = hif4_quantize_qkv(q_e, q_s, k_e, k_s, st)
"""
import torch

_U = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_SIGNED = torch.cat([_U, -_U])

_E6M2 = torch.zeros(256)
for _b in range(255):
    _E6M2[_b] = float(2.0 ** (((_b >> 2) & 63) - 48) * (1 + (_b & 3) / 4))
SCALE_TABLE = torch.tensor(sorted(set(float(_E6M2[b]) for b in range(255))))
_N = len(SCALE_TABLE)
_OFFSETS = torch.tensor([-1, 0, 1, 2, 3])
_MULTS = torch.tensor([1.0, 2.0, 4.0])


def hif4_decode(sf, s2, s3, sign, mant, dim=-1):
    p = sf * s2 * s3 * sign * mant
    d = dim % (p.ndim - 3); sh = list(p.shape)
    return p.reshape(*sh[:d], sh[d]*sh[d+1]*sh[d+2]*sh[d+3], *sh[d+4:])


# ── internals ────────────────────────────────────────────────────────

def _nv_mul(e, s):
    return (e.float().reshape(*e.shape[:-1], -1, 16)
          * s.float().reshape(*s.shape[:-1], -1, 1)).flatten(-2, -1)


def _cat_nv(cal_list, C):
    return torch.cat([_nv_mul(e, s).reshape(-1, C) for e, s in cal_list], 0)


def _encode(target, chunk=512):
    B = target.shape[0]; sgn = torch.where(target >= 0, 1., -1.)
    sf_o=torch.zeros(B); ki_o=torch.zeros(B,64,dtype=torch.long)
    e8_o=torch.zeros(B,8,dtype=torch.long); e16_o=torch.zeros(B,16,dtype=torch.long)
    for c0 in range(0, B, chunk):
        c1=min(c0+chunk,B); n=c1-c0
        blk=target[c0:c1]; s=sgn[c0:c1]; mag=blk.abs().reshape(n,16,4)
        g8=mag.max(2).values.reshape(n,8,2).max(2).values
        ci=torch.searchsorted(SCALE_TABLE,g8.max(1).values/7).clamp(0,_N-1)
        cands=SCALE_TABLE[(ci[:,None]+_OFFSETS).clamp(0,_N-1)]; SC=5
        inv4=4.0/(cands[:,:,None]*_MULTS); esq16=1.0/(inv4*inv4*16)
        err3=torch.empty(n,SC,8,2,3); km=[]
        for mi in range(3):
            q4=mag[:,None,:,:]*inv4[:,:,mi:mi+1,None]
            k=(q4+.5-1e-9).int().clamp(0,7)
            err3[:,:,:,:,mi]=((q4-k.float()).pow(2).sum(3)*esq16[:,:,mi:mi+1]).reshape(n,SC,8,2)
            km.append(k)
        p0=err3[:,:,:,:,1]<err3[:,:,:,:,0]; p1=err3[:,:,:,:,2]<err3[:,:,:,:,1]
        c0_=torch.where(p0,err3[:,:,:,:,1],err3[:,:,:,:,0]).sum(3)
        c1_=torch.where(p1,err3[:,:,:,:,2],err3[:,:,:,:,1]).sum(3)
        u=c1_<c0_; be=u.long(); bm=torch.where(u[:,:,:,None],p1.long(),p0.long())
        midx=be.repeat_interleave(8,2)+bm.reshape(n,SC,16).repeat_interleave(4,2)
        k0,k1,k2=(m.reshape(n,SC,64) for m in km)
        bk=torch.where(midx==0,k0,torch.where(midx==1,k1,k2)).long()
        ec=bk.float()*(0.25*(1<<midx).float()); ab=blk.abs()
        cjso=(ab[:,None,:]*ec).sum(2)/(ec*ec).sum(2).clamp(min=1e-30)
        ci2=torch.searchsorted(SCALE_TABLE,cjso.reshape(-1)).clamp(0,_N-1).reshape(n,SC)
        cl=(ci2-1).clamp(0)
        bsf=torch.where((SCALE_TABLE[ci2]-cjso).abs()<(SCALE_TABLE[cl]-cjso).abs(),
                         SCALE_TABLE[ci2],SCALE_TABLE[cl])
        recon=s[:,None,:]*ec*bsf[:,:,None]
        mse=(blk[:,None,:]-recon).pow(2).sum(2)
        w=mse.argmin(1); bi=torch.arange(n)
        sf_o[c0:c1]=bsf[bi,w]; ki_o[c0:c1]=bk[bi,w]
        e8_o[c0:c1]=be[bi,w]; e16_o[c0:c1]=bm[bi,w].reshape(n,16)
    return sf_o, e8_o, e16_o, ki_o, sgn


def _errdiff(target, svd, n_refine=3):
    """Calibration-aware floor/ceil selection in two phases:

    Phase 1 — GPTQ-like greedy pass (importance-ordered):
      Process columns in decreasing activation importance.
      Important columns see ev≈0 → protected by round-to-nearest.
      Unimportant columns absorb accumulated error via flips.

    Phase 2 — Ritz coordinate-descent refinement:
      Re-examine every element's decision using the GLOBAL error state.
      Flip any element where the change reduces the cost function.
      Repeat until convergence (typically 1-3 sweeps, <10% overhead).

    Cost function (same in both phases):
      cost(d) = d² × h_c  +  2·d·dot·conf_c
    """
    B = target.shape[0]
    vw = svd["vw"]; an_full = svd["an_full"]; an_svd = svd["an_svd"]
    C = vw.shape[1]; k = vw.shape[0]; nb = C // 64; R = B // nb

    sf, e8, e16, ki_std, sgn = _encode(target)
    mv = 2.0**(e8.repeat_interleave(8,1)+torch.stack(
        [e16.reshape(B,8,2)[:,:,0],e16.reshape(B,8,2)[:,:,1]],2
    ).reshape(B,16).repeat_interleave(4,1)).float()
    eff = sf[:, None] * mv

    q4 = target.abs() / eff.clamp(min=1e-30) * 4
    kl = q4.long().clamp(0, 7); kh = (kl + 1).clamp(0, 7)
    ki_alt = torch.where(ki_std == kl, kh, kl)
    d_std = target - sgn * ki_std.float() * 0.25 * eff
    d_alt = target - sgn * ki_alt.float() * 0.25 * eff

    # ── Phase 1: GPTQ greedy pass ────────────────────────────
    ev = torch.zeros(R, k); ki_ed = ki_std.clone()
    order = an_full.argsort(descending=True)

    for col in order:
        b = col // 64; j = col % 64
        bi = torch.arange(R) * nb + b
        vj = vw[:, col]
        dot = ev @ vj
        anj = an_full[col]
        conf = an_svd[col] / anj.clamp(min=1e-30)
        c_s = d_std[bi, j] ** 2 * anj + 2 * d_std[bi, j] * dot * conf
        c_a = d_alt[bi, j] ** 2 * anj + 2 * d_alt[bi, j] * dot * conf
        flip = c_a < c_s
        ki_ed[bi, j] = torch.where(flip, ki_alt[bi, j], ki_std[bi, j])
        delta = torch.where(flip, d_alt[bi, j], d_std[bi, j])
        ev = ev + delta[:, None] * vj[None, :]

    if n_refine <= 0:
        return sf, e8, e16, ki_ed, sgn

    # ── Phase 2: Ritz block-coordinate refinement ────────────
    d_cur = target - sgn * ki_ed.float() * 0.25 * eff
    vj_sq = (vw ** 2).sum(0)                               # (C,)

    # Build y = total projected error from Phase 1 result
    y = torch.zeros(R, k)
    for b in range(nb):
        bi = torch.arange(R) * nb + b
        y = y + d_cur[bi] @ vw[:, b*64:(b+1)*64].T

    for _it in range(n_refine):
        total_flips = 0
        for b in range(nb):
            bi = torch.arange(R) * nb + b
            sl = slice(b * 64, (b + 1) * 64)
            vw_b = vw[:, sl]                                # (k, 64)
            h_b = an_full[sl]; conf_b = an_svd[sl] / h_b.clamp(min=1e-30)

            # Remove self-contribution: dot = y·vj - d_cur·||vj||²
            dot = y @ vw_b - d_cur[bi] * vj_sq[sl][None, :]

            c_s = d_std[bi]**2 * h_b[None,:] + 2 * d_std[bi] * dot * conf_b[None,:]
            c_a = d_alt[bi]**2 * h_b[None,:] + 2 * d_alt[bi] * dot * conf_b[None,:]
            should_alt = c_a < c_s

            old_alt = (ki_ed[bi] == ki_alt[bi])
            changed = should_alt != old_alt
            n_ch = changed.sum().item()
            total_flips += n_ch
            if n_ch > 0:
                new_ki = torch.where(should_alt, ki_alt[bi], ki_std[bi])
                new_d = torch.where(should_alt, d_alt[bi], d_std[bi])
                y = y + (new_d - d_cur[bi]) @ vw_b.T
                d_cur[bi] = new_d
                ki_ed[bi] = new_ki

        if total_flips == 0:
            break

    return sf, e8, e16, ki_ed, sgn

def _svd_cal(A_flat, nb, rank=32):
    """Global SVD of calibration matrix with soft-threshold shrinkage.
    Debiases singular values when N is small (Marchenko-Pastur regime)."""
    assert nb > 0 and rank > 0 and A_flat.shape[0] > 0
    N, C = A_flat.shape; k = min(rank, N, C)
    _, S, Vh = torch.linalg.svd(A_flat, full_matrices=False)
    V_k = Vh[:k]; S_k = S[:k].clamp(min=0)
    # Soft-threshold: only when few samples (N < 4k)
    if S.shape[0] > k and N < 4 * k:
        noise = S[k:].pow(2).mean().sqrt()
        S_k = (S_k - noise).clamp(min=0)
    vw = V_k * S_k[:, None]                              # (k, C)
    an_full = (A_flat ** 2).sum(0)                        # (C,)
    an_svd = (vw ** 2).sum(0)                             # (C,)
    return {"vw": vw, "an_full": an_full, "an_svd": an_svd}


def _svd_cal_streaming(stream_fn, D, nb, rank=32):
    """2-pass Krylov subspace iteration — streaming SVD replacement.

    Never stores the N×D calibration matrix. State: O(D × rank).

    Pass 1: accumulate H×Ω (random probe) + an_full
    Pass 2: refine subspace via H×Q, then eigendecompose Q^T H Q

    Args:
        stream_fn: callable that yields (B, D) activation batches.
                   Called twice (two passes through calibration data).
        D: activation dimension
        nb: number of 64-element blocks (D // 64)
        rank: number of SVD directions to compute
    """
    k = rank
    torch.manual_seed(0)
    Q = torch.randn(D, k)

    an_full = torch.zeros(D)
    for p in range(2):
        HQ = torch.zeros(D, k)
        for x in stream_fn():
            if x.dim() == 1: x = x.unsqueeze(0)
            if p == 0: an_full += (x ** 2).sum(0)
            HQ += x.T @ (x @ Q)
        Q, _ = torch.linalg.qr(HQ)

    Q = Q[:, :k]
    # Eigenvalues via one more pass: Q^T H Q
    QtHQ = torch.zeros(k, k)
    for x in stream_fn():
        if x.dim() == 1: x = x.unsqueeze(0)
        qx = x @ Q
        QtHQ += qx.T @ qx

    eigvals, eigvecs = torch.linalg.eigh(QtHQ)
    idx = eigvals.argsort(descending=True)
    S = eigvals[idx].clamp(min=0).sqrt()
    V = (Q @ eigvecs[:, idx]).T                            # (k, D)

    # Soft-threshold (same as _svd_cal)
    N_est = an_full.sum() / an_full.mean().clamp(min=1e-30)
    if S.shape[0] > k and N_est < 4 * k:
        noise = S[k:].pow(2).mean().sqrt() if S.shape[0] > k else 0
        S = (S[:k] - noise).clamp(min=0)
    else:
        S = S[:k]

    vw = V[:k] * S[:k, None]
    an_svd = (vw ** 2).sum(0)
    return {"vw": vw, "an_full": an_full, "an_svd": an_svd}


def _to_blocks(nv_e, nv_s, dim=-1):
    d = dim % nv_e.ndim; C = nv_e.shape[d]; assert C % 64 == 0
    perm = list(range(nv_e.ndim)); perm.append(perm.pop(d))
    ne = nv_e.permute(*perm).contiguous()
    ns = nv_s.float().permute(*perm).contiguous()
    prefix = ne.shape[:-1]; nb = C // 64
    nr = 1
    for p in prefix: nr *= p
    B = nr * nb
    target = (ne.float().reshape(B, 4, 16) * ns.reshape(B, 4, 1)).reshape(B, 64)
    return target, prefix, nb, d


def _pack(sf, e8, e16, ki, sgn, prefix, nb, d):
    B = sf.shape[0]
    def shaped(t, *bd):
        t = t.reshape(*prefix, nb, *bd)
        src = list(range(len(prefix), len(prefix)+1+len(bd)))
        dst = list(range(d, d+1+len(bd)))
        return torch.movedim(t, src, dst).contiguous()
    return (shaped(sf.reshape(B,1,1,1),                                1,1,1),
            shaped((1<<e8).float().reshape(B,8,1,1),                   8,1,1),
            shaped((1<<e16.reshape(B,8,2)).float().reshape(B,8,2,1),   8,2,1),
            shaped(sgn.reshape(B,8,2,4),                               8,2,4),
            shaped(ki.float().reshape(B,8,2,4)*0.25,                   8,2,4))


@torch.jit.script
def _schwarz_loop(y: torch.Tensor, dc: torch.Tensor,
                  grp_cols: torch.Tensor,
                  vw_a: torch.Tensor, gr_a: torch.Tensor, gh_a: torch.Tensor,
                  ds_a: torch.Tensor, dd_a: torch.Tensor, da_a: torch.Tensor,
                  pats: torch.Tensor, bits: torch.Tensor,
                  n_iter: int, P: int, G: int) -> torch.Tensor:
    """JIT-compiled Schwarz inner loop — eliminates Python dispatch overhead.

    Processes groups sequentially (Gauss-Seidel), updating y after each group.
    Each group: enumerate all 2^G patterns via Gram trick, pick best per row.
    """
    NG = grp_cols.shape[0]
    R = dc.shape[0]
    for _it in range(n_iter):
        for gi in range(NG):
            cols = grp_cols[gi]
            d_c = dc[:, cols]                                      # (R, G)
            dot_wo = torch.mm(y, vw_a[gi]) - torch.mm(d_c, gr_a[gi])
            dt = ds_a[gi].unsqueeze(0) + pats * dd_a[gi].unsqueeze(0)
            quad = torch.mm(dt.reshape(P * R, G), gh_a[gi]).reshape(P, R, G)
            cost = (dt * (dot_wo.unsqueeze(0) * 2 + quad)).sum(2)
            bp = cost.argmin(0)                                    # (R,)
            bp_bits = ((bp[:, None] >> bits[None, :]) & 1) == 1
            new_d = torch.where(bp_bits, da_a[gi], ds_a[gi])
            delta = new_d - d_c
            dc[:, cols] = new_d
            y = y + torch.mm(delta, vw_a[gi].T)
    return y


def _schwarz(target, svd, G=4, n_iter=3, overlap=0):
    """Schwarz alternating domain decomposition for HiF4 quantization.

    Decomposes columns into importance-ordered non-overlapping groups
    of G, solves each group EXACTLY (enumerates all 2^G floor/ceil
    combinations), and iterates until convergence (fixed point).

    Uses the Gram trick: cost is computed via the G×G matrix
    gram_h = vw^T vw + diag(h), avoiding the (P,R,k) projection
    tensor.  All constants are pre-gathered and contiguous.

    Compared to GPTQ (_errdiff):
      GPTQ:    greedy sequential, rmse ~1.33×, growth 1.095/layer
      Schwarz: exact groups, iterative, rmse ~1.11×, growth 1.072/layer
    """
    B = target.shape[0]
    vw = svd["vw"]; an_f = svd["an_full"]
    C = vw.shape[1]; k = vw.shape[0]; nb = C // 64; R = B // nb

    sf, e8, e16, ki_std, sgn = _encode(target)
    mv = 2.0**(e8.repeat_interleave(8, 1) + torch.stack(
        [e16.reshape(B, 8, 2)[:, :, 0], e16.reshape(B, 8, 2)[:, :, 1]], 2
    ).reshape(B, 16).repeat_interleave(4, 1)).float()
    eff = sf[:, None] * mv

    # ── Build global error tables ────────────────────────────
    d_std_g = torch.zeros(R, C)
    d_alt_g = torch.zeros(R, C)
    ki_alt_g = torch.zeros(R, C, dtype=torch.long)
    for b in range(nb):
        bi = torch.arange(R) * nb + b
        q4 = target[bi].abs() / eff[bi].clamp(min=1e-30) * 4
        kl = q4.long().clamp(0, 7); kh = (kl + 1).clamp(0, 7)
        ka = torch.where(ki_std[bi] == kl, kh, kl)
        d_std_g[:, b*64:(b+1)*64] = target[bi] - sgn[bi] * ki_std[bi].float() * 0.25 * eff[bi]
        d_alt_g[:, b*64:(b+1)*64] = target[bi] - sgn[bi] * ka.float() * 0.25 * eff[bi]
        ki_alt_g[:, b*64:(b+1)*64] = ka

    d_diff = d_alt_g - d_std_g
    d_cur = d_std_g.clone()
    y = d_cur @ vw.T                                              # (R, k)

    # ── Pre-compute group structure (one-time) ───────────────
    order = an_f.argsort(descending=True)
    stride = max(1, G - overlap)
    NG = C // stride if overlap == 0 else \
         sum(1 for s in range(0, C, stride) if s + G <= C)
    grp_cols = order[:NG * G].reshape(NG, G) if overlap == 0 else \
               torch.stack([order[s:s+G] for s in range(0, C, stride)
                            if order[s:s+G].shape[0] == G])
    NG = grp_cols.shape[0]
    flat = grp_cols.reshape(-1)

    vw_a = vw[:, flat].reshape(k, NG, G).permute(1, 0, 2).contiguous()
    h_a = an_f[flat].reshape(NG, G)
    gr_a = torch.bmm(vw_a.permute(0, 2, 1), vw_a).contiguous()
    gh_a = (gr_a + torch.diag_embed(h_a)).contiguous()

    # Pre-gather constant group data
    d_s_a = d_std_g[:, flat].reshape(R, NG, G).permute(1, 0, 2).contiguous()
    d_d_a = d_diff[:, flat].reshape(R, NG, G).permute(1, 0, 2).contiguous()
    d_alt_a = d_alt_g[:, flat].reshape(R, NG, G).permute(1, 0, 2).contiguous()

    P = 2 ** G
    pats = ((torch.arange(P)[:, None] >> torch.arange(G)[None, :]) & 1).float()
    pats = pats.view(P, 1, G)                                     # (P, 1, G)
    bits = torch.arange(G)

    # ── Schwarz iteration (JIT-compiled inner loop) ──────────
    y = _schwarz_loop(y, d_cur, grp_cols, vw_a, gr_a, gh_a,
                      d_s_a, d_d_a, d_alt_a, pats, bits, n_iter, P, G)

    # ── Write back to block layout ───────────────────────────
    ki_ed = ki_std.clone()
    for b in range(nb):
        bi = torch.arange(R) * nb + b
        is_alt = (d_cur[:, b*64:(b+1)*64] - d_std_g[:, b*64:(b+1)*64]).abs() > 1e-30
        ki_ed[bi] = torch.where(is_alt, ki_alt_g[:, b*64:(b+1)*64], ki_std[bi])

    return sf, e8, e16, ki_ed, sgn


def _quantize(nv_e, nv_s, svd=None, dim=-1, method='schwarz'):
    target, prefix, nb, d = _to_blocks(nv_e, nv_s, dim)
    if svd is not None:
        if method == 'schwarz':
            sf, e8, e16, ki, sgn = _schwarz(target, svd)
        else:
            sf, e8, e16, ki, sgn = _errdiff(target, svd, n_refine=0)
    else:
        sf, e8, e16, ki, sgn = _encode(target)
    return _pack(sf, e8, e16, ki, sgn, prefix, nb, d)


# ── public: no calibration ───────────────────────────────────────────

def nvfp4_to_hif4(nv_elements, nv_scales, dim=-1):
    return _quantize(nv_elements, nv_scales, dim=dim)


# ── public: W @ A^T ──────────────────────────────────────────────────

def hif4_calibration_and_quantize_weight(weight_quant, weight_scale,
                                         calib_activation_list,
                                         rank=32, dim=-1):
    """Calibrate and quantize NVFP4 weight to HiF4 using activation statistics.
    Offline only — run once per layer during model quantization.
    """
    C = weight_quant.shape[-1]; nb = C // 64
    A = _cat_nv(calib_activation_list, C)
    svd_a = _svd_cal(A, nb, rank)
    return _quantize(weight_quant, weight_scale, svd_a, dim)


# ── public: QKV ──────────────────────────────────────────────────────

def hif4_calibration_attention(cal_samples, q_heads, kv_heads, head_dim, rank=32):
    assert q_heads > 0 and kv_heads > 0 and head_dim > 0
    assert q_heads % kv_heads == 0
    hpk = q_heads // kv_heads
    D_q = q_heads * head_dim; D_kv = kv_heads * head_dim

    Q = torch.cat([_nv_mul(*s["q"]).reshape(-1, D_q)  for s in cal_samples], 0)
    K = torch.cat([_nv_mul(*s["k"]).reshape(-1, D_kv) for s in cal_samples], 0)

    K_exp = K.reshape(-1, kv_heads, head_dim).repeat_interleave(hpk, 1).reshape(-1, D_q)
    Q_red = Q.reshape(-1, kv_heads, hpk, head_dim).mean(2).reshape(-1, D_kv)

    state = {"q_state": _svd_cal(K_exp, D_q // 64, rank),
             "k_state": _svd_cal(Q_red, D_kv // 64, rank)}

    if any("v" in s for s in cal_samples):
        V = torch.cat([_nv_mul(*s["v"]).reshape(-1, D_kv)
                       for s in cal_samples if "v" in s], 0)
        state["v_state"] = _svd_cal(V, D_kv // 64, rank)

    return state


def hif4_quantize_qkv(q_elem, q_scale, k_elem, k_scale,
                       state, v_elem=None, v_scale=None, dim=-1):
    q = _quantize(q_elem, q_scale, state["q_state"], dim)
    k = _quantize(k_elem, k_scale, state["k_state"], dim)
    v = _quantize(v_elem, v_scale, state["v_state"], dim) \
        if v_elem is not None and "v_state" in state else None
    return q, k, v


# ── test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    def nrmse(a, b):
        return (a-b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()

    def make_nvfp4(fp_data):
        D = fp_data.shape[-1]; flat = fp_data.reshape(-1, D//16, 16)
        s = flat.abs().max(2).values.div(6).clamp(min=1e-6)
        normed = flat / s.unsqueeze(-1)
        codes = _U[(normed.abs().unsqueeze(-1) - _U).abs().argmin(-1)]
        return (codes * normed.sign()).reshape(*fp_data.shape), s

    def make_llm_data(D, R, N_cal, N_test, noise=0.1):
        """Realistic LLM-like data with shared embedding basis."""
        rank = max(4, D // 8)
        ch = torch.exp(torch.randn(D) * 0.5)
        basis = torch.randn(rank, D) * ch[None, :]
        W_fp = torch.randn(R, D) * 0.02
        cal = torch.randn(N_cal, rank) @ basis / rank**.5 \
            + torch.randn(N_cal, D) * ch * noise
        test = torch.randn(N_test, rank) @ basis / rank**.5 \
             + torch.randn(N_test, D) * ch * noise
        return W_fp, cal, test

    # ── W @ A^T ──
    print("W @ A^T"); print("=" * 60)
    print(f"  {'':20s} {'score':>9s}")
    for label, D, R, noise in [
        ("D=256 noise=0.1",  256,  64, 0.1),
        ("D=256 noise=0.5",  256,  64, 0.5),
        ("D=256 noise=1.0",  256,  64, 1.0),
        ("D=512 noise=0.1",  512, 128, 0.1),
        ("D=512 noise=0.5",  512, 128, 0.5),
        ("D=1024 noise=0.1", 1024, 256, 0.1),
    ]:
        torch.manual_seed(42)
        W_fp, cal_fp, test_fp = make_llm_data(D, R, 128, 64, noise)
        We, Ws = make_nvfp4(W_fp); W = _nv_mul(We, Ws)
        cal_list = [make_nvfp4(cal_fp[i:i+32]) for i in range(0, 128, 32)]
        Y_true = W @ test_fp.T
        Wi_std = hif4_decode(*nvfp4_to_hif4(We, Ws)); Y_std = Wi_std @ test_fp.T
        cal_params = hif4_calibration_and_quantize_weight(We, Ws, cal_list)
        Wi_cal = hif4_decode(*cal_params); Y_cal = Wi_cal @ test_fp.T
        bm = (Y_true-Y_std).pow(2).mean(); am = (Y_true-Y_cal).pow(2).mean()
        yg = ((bm-am)/bm*100).item()
        print(f"  {label:20s} {yg:+8.1f}%")

    # ── QKV ──
    print(f"\nQKV attention"); print("=" * 60)
    print(f"  {'':20s} {'score_nrmse':>12s}")
    for label, D_model, qH, kvH, hd, noise in [
        ("8h/64d noise=0.1",  512,  8, 8,  64, 0.1),
        ("8h/64d noise=0.5",  512,  8, 8,  64, 0.5),
        ("32h/8kv noise=0.1", 1024, 32, 8, 128, 0.1),
    ]:
        torch.manual_seed(42)
        D_q = qH*hd; D_kv = kvH*hd; hpk = qH//kvH
        rank = max(4, D_model//8)
        ch = torch.exp(torch.randn(D_model) * 0.5)
        basis = torch.randn(rank, D_model) * ch[None, :]
        W_q = torch.randn(D_q, D_model) * 0.02
        W_k = torch.randn(D_kv, D_model) * 0.02
        cal_samples = []
        for _ in range(4):
            X = torch.randn(32, rank) @ basis / rank**.5 \
              + torch.randn(32, D_model) * ch * noise
            cal_samples.append({"q": make_nvfp4(X @ W_q.T),
                                "k": make_nvfp4(X @ W_k.T)})
        st = hif4_calibration_attention(cal_samples, qH, kvH, hd)
        X_test = torch.randn(16, rank) @ basis / rank**.5 \
               + torch.randn(16, D_model) * ch * noise
        Qe, Qs = make_nvfp4(X_test @ W_q.T)
        Ke, Ks = make_nvfp4(X_test @ W_k.T)
        Q_nv = _nv_mul(Qe, Qs); K_nv = _nv_mul(Ke, Ks)
        K_exp = K_nv.reshape(-1, kvH, hd).repeat_interleave(hpk, 1).reshape(-1, D_q)
        sc_true = Q_nv @ K_exp.T / hd**.5
        Qi = hif4_decode(*nvfp4_to_hif4(Qe, Qs))
        Ki = hif4_decode(*nvfp4_to_hif4(Ke, Ks))
        Ki_e = Ki.reshape(-1, kvH, hd).repeat_interleave(hpk, 1).reshape(-1, D_q)
        sc_std = Qi @ Ki_e.T / hd**.5
        q_h, k_h, _ = hif4_quantize_qkv(Qe, Qs, Ke, Ks, st)
        Qc = hif4_decode(*q_h); Kc = hif4_decode(*k_h)
        Kc_e = Kc.reshape(-1, kvH, hd).repeat_interleave(hpk, 1).reshape(-1, D_q)
        sc_cal = Qc @ Kc_e.T / hd**.5
        g = (1 - nrmse(sc_true, sc_cal) / nrmse(sc_true, sc_std)) * 100
        print(f"  {label:20s} {g:+11.1f}%")

    # ── Speed ──
    print(f"\nSpeed"); print("=" * 60)
    for D, R in [(256, 64), (512, 128), (1024, 256)]:
        torch.manual_seed(0)
        We, Ws = make_nvfp4(torch.randn(R, D) * 0.02)
        cl = [make_nvfp4(torch.randn(32, D) * 0.5) for _ in range(4)]
        for nm, fn in [("standard", lambda: nvfp4_to_hif4(We, Ws)),
                       ("calibrated", lambda: hif4_calibration_and_quantize_weight(We, Ws, cl))]:
            for _ in range(3): fn()
            t0 = time.time()
            for _ in range(10): fn()
            print(f"  D={D:4d} {nm:12s}: {(time.time()-t0)/10*1000:.0f}ms")
