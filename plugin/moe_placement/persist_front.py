"""Front and finalize kernels of the host-aware persistent MoE (Triton).

front     one launch, P programs. Every program recomputes the routing record v2 (route_ref.py) from the top-k ids in
          registers (histogram + scans over 384 experts: cheap), so no inter-program sync is needed; program 0 writes
          the record tables and the per-GEMM schedule (kc, hoc), every program writes `permuted` and gathers the MXFP8
          rows (and scales, into the per-tile atom layout GEMM1's B-scale TMA reads) of its own slots into xp.
finalize  out[t] = sum_k w[t, k] * rows[permuted[t, k]] in fp32, ascending k, -> bf16 (for calls that do not defer).
Both are plain launches on the current stream (CUDA-graph capturable); GEMM1 (CuTe, PDL) waits on the front with
griddepcontrol.wait, so only its prologue overlaps the front.
CPU test: TRITON_INTERPRET=1 (test_front.py).
"""
import os

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

# PDL: the front waits for its predecessor with griddepcontrol.wait and lets GEMM1 launch at once (GEMM1 waits for the
# whole front before it reads the record); finalize waits for GEMM2 the same way. MOE_PERSIST_PDL=0 disables.
PDL = os.environ.get("MOE_PERSIST_PDL", "1") == "1"
# front kernel version: 2 = loads first + one packed scan (default), 1 = the first version
FRONT = int(os.environ.get("MOE_PERSIST_FRONT", "2"))

E_DEF, K_DEF, HID_DEF, SFK_DEF = 384, 6, 5120, 160
HDR = tl.constexpr(12)
ITEM_BYTES = (128 * 2560, 128 * 1152)
M_TILES = (36, 40)


@triton.jit
def _front_kernel(ids_ptr, S, xq_ptr, xsf_ptr, host_ptr,
                  rec_ptr, perm_ptr, xp_ptr, xpsf_ptr,
                  C, G, host_bw, hbm_bw, cta_bw, ovh,
                  K: tl.constexpr, E: tl.constexpr, EB: tl.constexpr, SB: tl.constexpr, MYB: tl.constexpr,
                  HID: tl.constexpr, HB: tl.constexpr, SFK: tl.constexpr, SFB: tl.constexpr, JMAX: tl.constexpr,
                  KC_MIN: tl.constexpr, KC_MAX: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        gdc_launch_dependents()
        gdc_wait()
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    # ---- the whole routing, recomputed by every program
    s_all = tl.arange(0, SB)
    m_all = s_all < S
    ids = tl.load(ids_ptr + s_all, mask=m_all, other=-1)
    ok = m_all & (ids >= 0) & (ids < E)
    cnt = tl.histogram(tl.where(ok, ids, 0), EB, mask=ok)
    e = tl.arange(0, EB)
    ntile = (cnt + 7) // 8
    ftile = tl.cumsum(ntile, 0) - ntile
    fcrow = tl.cumsum(cnt, 0) - cnt
    tiles = tl.sum(ntile, 0)
    hm = tl.load(host_ptr + e, mask=e < E, other=0) != 0
    ncold_e = tl.where(hm, ntile, 0)
    n_cold = tl.sum(ncold_e, 0)
    nhot_e = ntile - ncold_e
    pos0 = tl.where(hm, tl.cumsum(ncold_e, 0) - ncold_e, tl.cumsum(nhot_e, 0) - nhot_e + n_cold)

    # ---- this program's slots: rank among earlier slots of the same expert (tokens ascending)
    my = pid + P * tl.arange(0, MYB)
    mm = my < S
    myid = tl.load(ids_ptr + my, mask=mm, other=-1)
    myok = mm & (myid >= 0) & (myid < E)
    eq = (ids[None, :] == myid[:, None]) & (s_all[None, :] < my[:, None]) & ok[None, :]
    rank = tl.sum(eq.to(tl.int32), 1)
    oh = e[None, :] == myid[:, None]
    ft = tl.sum(tl.where(oh, ftile[None, :], 0), 1)
    fc = tl.sum(tl.where(oh, fcrow[None, :], 0), 1)
    prow = ft * 8 + rank
    tl.store(perm_ptr + my, tl.where(myok, fc + rank, -1), mask=mm)
    tok = (my // K).to(tl.int64)
    prow64 = prow.to(tl.int64)
    for j in tl.static_range(0, HID, HB):
        cols = j + tl.arange(0, HB)
        v = tl.load(xq_ptr + tok[:, None] * HID + cols[None, :], mask=myok[:, None], other=0)
        tl.store(xp_ptr + prow64[:, None] * HID + cols[None, :], v, mask=myok[:, None])
    jj = tl.arange(0, SFB)
    mj = myok[:, None] & (jj[None, :] < SFK)
    sv = tl.load(xsf_ptr + tok[:, None] * SFK + jj[None, :], mask=mj, other=0)
    dst = (prow64 // 8)[:, None] * (SFK // 4 * 512) + (jj // 4)[None, :] * 512 + (prow64 % 8)[:, None] * 16 + (jj % 4)[None, :]
    tl.store(xpsf_ptr + dst, sv, mask=mj)

    # ---- program 0: record header, tile tables, cold-first order, schedule
    if pid == 0:
        tl.store(rec_ptr + 0, tiles * 8)
        tl.store(rec_ptr + 1, tiles)
        tl.store(rec_ptr + 2, n_cold)
        tl.store(rec_ptr + 3, S // K)
        for j in tl.static_range(JMAX):
            okj = (e < E) & (j < ntile)
            tile = ftile + j
            tl.store(rec_ptr + HDR + tile, e, mask=okj)
            tl.store(rec_ptr + HDR + C + tile, ftile * 8 + tl.minimum(cnt, (j + 1) * 8), mask=okj)
            tl.store(rec_ptr + HDR + 2 * C + tile, fcrow + j * 8, mask=okj)
            tl.store(rec_ptr + HDR + 3 * C + pos0 + j, tile, mask=okj)
        for g in tl.static_range(2):
            if g == 0:
                mt = 36
                nb = 327680.0
            else:
                mt = 40
                nb = 147456.0
            nci = n_cold * mt
            nhi = (tiles - n_cold) * mt
            best_t = 1.0e30
            best_kc = 0
            best_hoc = 0
            for kc in tl.static_range(KC_MIN, KC_MAX + 1):
                t_cold = nb / tl.minimum(cta_bw, host_bw / kc) + ovh
                t_hot = nb / tl.minimum(cta_bw, hbm_bw / (G - kc)) + ovh
                tc = ((nci + kc - 1) // kc).to(tl.float32) * t_cold
                th = nhi.to(tl.float32) * t_hot / (G - kc)
                t_join = (nhi.to(tl.float32) * t_hot + kc * tc) / G
                hoc = tl.floor(kc * (t_join - tc) / t_hot + 0.5).to(tl.int32)
                hoc = tl.minimum(tl.maximum(hoc, 0), nhi)
                tm = tl.where(th <= tc, tc, t_join)
                hoc = tl.where(th <= tc, 0, hoc)
                better = tm < best_t * (1.0 - 1e-7)
                best_t = tl.where(better, tm, best_t)
                best_kc = tl.where(better, kc, best_kc)
                best_hoc = tl.where(better, hoc, best_hoc)
            best_kc = tl.where(nci == 0, 0, best_kc)
            best_hoc = tl.where(nci == 0, 0, best_hoc)
            tl.store(rec_ptr + 4 + 4 * g, best_kc)
            tl.store(rec_ptr + 5 + 4 * g, best_hoc)


@triton.jit
def _front_kernel2(ids_ptr, S, xq_ptr, xsf_ptr, host_ptr,
                   rec_ptr, perm_ptr, xp_ptr, xpsf_ptr,
                   C, G, host_bw, hbm_bw, cta_bw, ovh,
                   K: tl.constexpr, E: tl.constexpr, EB: tl.constexpr, SB: tl.constexpr, MYB: tl.constexpr,
                   HID: tl.constexpr, HB: tl.constexpr, SFK: tl.constexpr, SFB: tl.constexpr, JMAX: tl.constexpr,
                   KC_MIN: tl.constexpr, KC_MAX: tl.constexpr, USE_PDL: tl.constexpr):
    """Same outputs as _front_kernel, shorter critical path: every global load (all ids, the host mask, this program's
    x rows and scales -- their source rows are known from the slot index alone) is issued right after the PDL wait,
    and the three prefix sums over experts are one scan of packed counts (cnt | tiles << 11 | cold tiles << 21)."""
    if USE_PDL:
        gdc_launch_dependents()
        gdc_wait()
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    s_all = tl.arange(0, SB)
    m_all = s_all < S
    e = tl.arange(0, EB)
    my = pid + P * tl.arange(0, MYB)
    mm = my < S
    tok = (my // K).to(tl.int64)
    cols = tl.arange(0, HB)
    jj = tl.arange(0, SFB)
    # ---- issue every load first
    ids = tl.load(ids_ptr + s_all, mask=m_all, other=-1)
    myid = tl.load(ids_ptr + my, mask=mm, other=-1)
    hm = tl.load(host_ptr + e, mask=e < E, other=0) != 0
    xv = tl.load(xq_ptr + tok[:, None] * HID + cols[None, :], mask=mm[:, None] & (cols[None, :] < HID), other=0)
    sv = tl.load(xsf_ptr + tok[:, None] * SFK + jj[None, :], mask=mm[:, None] & (jj[None, :] < SFK), other=0)
    # ---- routing (every program)
    ok = m_all & (ids >= 0) & (ids < E)
    cnt = tl.histogram(tl.where(ok, ids, 0), EB, mask=ok)
    ntile = (cnt + 7) // 8
    ncold_e = tl.where(hm, ntile, 0)
    pk = cnt | (ntile << 11) | (ncold_e << 21)
    tot = tl.sum(pk, 0)
    ex = tl.cumsum(pk, 0) - pk
    fcrow = ex & 2047
    ftile = (ex >> 11) & 1023
    cold_before = (ex >> 21) & 1023
    tiles = (tot >> 11) & 1023
    n_cold = (tot >> 21) & 1023
    pos0 = tl.where(hm, cold_before, ftile - cold_before + n_cold)
    myok = mm & (myid >= 0) & (myid < E)
    eq = (ids[None, :] == myid[:, None]) & (s_all[None, :] < my[:, None]) & ok[None, :]
    rank = tl.sum(eq.to(tl.int32), 1)
    oh = e[None, :] == myid[:, None]
    ft = tl.sum(tl.where(oh, ftile[None, :], 0), 1)
    fc = tl.sum(tl.where(oh, fcrow[None, :], 0), 1)
    prow64 = (ft * 8 + rank).to(tl.int64)
    tl.store(perm_ptr + my, tl.where(myok, fc + rank, -1), mask=mm)
    tl.store(xp_ptr + prow64[:, None] * HID + cols[None, :], xv, mask=myok[:, None] & (cols[None, :] < HID))
    dst = (prow64 // 8)[:, None] * (SFK // 4 * 512) + (jj // 4)[None, :] * 512 + (prow64 % 8)[:, None] * 16 + (jj % 4)[None, :]
    tl.store(xpsf_ptr + dst, sv, mask=myok[:, None] & (jj[None, :] < SFK))
    # ---- program 0: record header, tile tables, cold-first order, schedule
    if pid == 0:
        tl.store(rec_ptr + 0, tiles * 8)
        tl.store(rec_ptr + 1, tiles)
        tl.store(rec_ptr + 2, n_cold)
        tl.store(rec_ptr + 3, S // K)
        for j in tl.static_range(JMAX):
            okj = (e < E) & (j < ntile)
            tile = ftile + j
            tl.store(rec_ptr + HDR + tile, e, mask=okj)
            tl.store(rec_ptr + HDR + C + tile, ftile * 8 + tl.minimum(cnt, (j + 1) * 8), mask=okj)
            tl.store(rec_ptr + HDR + 2 * C + tile, fcrow + j * 8, mask=okj)
            tl.store(rec_ptr + HDR + 3 * C + pos0 + j, tile, mask=okj)
        for g in tl.static_range(2):
            if g == 0:
                mt = 36
                nb = 327680.0
            else:
                mt = 40
                nb = 147456.0
            nci = n_cold * mt
            nhi = (tiles - n_cold) * mt
            best_t = 1.0e30
            best_kc = 0
            best_hoc = 0
            for kc in tl.static_range(KC_MIN, KC_MAX + 1):
                t_cold = nb / tl.minimum(cta_bw, host_bw / kc) + ovh
                t_hot = nb / tl.minimum(cta_bw, hbm_bw / (G - kc)) + ovh
                tc = ((nci + kc - 1) // kc).to(tl.float32) * t_cold
                th = nhi.to(tl.float32) * t_hot / (G - kc)
                t_join = (nhi.to(tl.float32) * t_hot + kc * tc) / G
                hoc = tl.floor(kc * (t_join - tc) / t_hot + 0.5).to(tl.int32)
                hoc = tl.minimum(tl.maximum(hoc, 0), nhi)
                tm = tl.where(th <= tc, tc, t_join)
                hoc = tl.where(th <= tc, 0, hoc)
                better = tm < best_t * (1.0 - 1e-7)
                best_t = tl.where(better, tm, best_t)
                best_kc = tl.where(better, kc, best_kc)
                best_hoc = tl.where(better, hoc, best_hoc)
            best_kc = tl.where(nci == 0, 0, best_kc)
            best_hoc = tl.where(nci == 0, 0, best_hoc)
            tl.store(rec_ptr + 4 + 4 * g, best_kc)
            tl.store(rec_ptr + 5 + 4 * g, best_hoc)


@triton.jit
def _finalize_kernel(rows_ptr, w_ptr, perm_ptr, out_ptr, K: tl.constexpr, HID: tl.constexpr, BLOCK: tl.constexpr,
                     USE_PDL: tl.constexpr):
    if USE_PDL:
        gdc_wait()
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(K):
        p = tl.load(perm_ptr + t * K + k)
        w = tl.load(w_ptr + t * K + k).to(tl.float32)
        v = tl.load(rows_ptr + p.to(tl.int64) * HID + cols, mask=(cols < HID) & (p >= 0), other=0.0).to(tl.float32)
        acc += w * v
    tl.store(out_ptr + t * HID + cols, acc.to(tl.bfloat16), mask=cols < HID)


def max_tiles(T, K=K_DEF, E=E_DEF):
    slots = T * K
    filled = min(E, slots)
    return filled + (slots - filled) // 8


def cta_words(M):
    return (M + 3) // 4 * 4


def front(ids, xq, xsf, host_mask, rec, perm, xp, xpsf, G=152, host_bw=385e3, hbm_bw=6.0e6, cta_bw=46e3, ovh=0.25,
          programs=None, kc_min=4, kc_max=16):
    """ids [T, K] int32; xq [T, HID] uint8 (e4m3 bits); xsf [T, SFK] uint8; host_mask [E] int8; outputs preallocated:
    rec int32 [HDR + 4 C + 8 M], perm int32 [T, K], xp uint8 [8 M, HID], xpsf uint8 [M, SFK / 4 * 512]."""
    T, K = ids.shape
    S = T * K
    E = host_mask.shape[0]
    M = max_tiles(T, K, E)
    SB = triton.next_power_of_2(max(S, 16))
    if FRONT == 2:
        assert S <= 2047 and M <= 1023, (S, M)
        programs = max(1, -(-S // 4)) if programs is None else max(programs, -(-S // 8))   # <= 8 slots per program
        MYB = triton.next_power_of_2(max(1, -(-S // programs)))
        _front_kernel2[(programs,)](
            ids, S, xq, xsf, host_mask, rec, perm, xp, xpsf, cta_words(M), G, host_bw, hbm_bw, cta_bw, ovh,
            K=K, E=E, EB=triton.next_power_of_2(E), SB=SB, MYB=MYB, HID=xq.shape[1],
            HB=triton.next_power_of_2(xq.shape[1]), SFK=xsf.shape[1], SFB=triton.next_power_of_2(xsf.shape[1]),
            JMAX=triton.next_power_of_2(-(-T // 8)), KC_MIN=kc_min, KC_MAX=kc_max, USE_PDL=PDL,
            launch_pdl=PDL, num_warps=4 if MYB <= 4 else 8)
        return
    if programs is None:
        programs = max(1, min(128, -(-S // 3)))
    MYB = triton.next_power_of_2(max(1, -(-S // programs)))
    _front_kernel[(programs,)](
        ids, S, xq, xsf, host_mask, rec, perm, xp, xpsf, cta_words(M), G, host_bw, hbm_bw, cta_bw, ovh,
        K=K, E=E, EB=triton.next_power_of_2(E), SB=SB, MYB=MYB, HID=xq.shape[1], HB=1024, SFK=xsf.shape[1],
        SFB=triton.next_power_of_2(xsf.shape[1]), JMAX=-(-T // 8), KC_MIN=kc_min, KC_MAX=kc_max, USE_PDL=PDL,
        launch_pdl=PDL)


def finalize(rows, w, perm, out):
    T, K = perm.shape
    HID = out.shape[1]
    _finalize_kernel[(T, triton.cdiv(HID, 1024))](rows, w, perm, out, K=K, HID=HID, BLOCK=1024, USE_PDL=PDL,
                                                    launch_pdl=PDL)
