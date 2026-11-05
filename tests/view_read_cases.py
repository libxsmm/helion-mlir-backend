"""Kernels with a view of a tensor they write (Helion source, eager reference).

The semantics are ``HELION_INTERPRET=1``'s: a read sees the stores before it. Each
case: ``fn(x, ...)``, its block size count, a reference on pristine copies, and how
to make its arguments from ``x``."""

from __future__ import annotations

import helion.language as hl
import torch

M, N = 16, 32
CASES = {}


def _with_caller_view(x):
    a = x.clone()
    return x, a, a.view(2 * M, N // 2)


def case(name, n_blocks, ref, args=None, note=""):
    def deco(fn):
        CASES[name] = {
            "fn": fn,
            "blocks": n_blocks,
            "ref": ref,
            "args": args,
            "note": note,
        }
        return fn

    return deco


# --- written phase 1, read phase 2 (the bug)
@case("A_w1_r2_reshape", 4, lambda x: (x * 2).view(2 * M, N // 2) + 1)
def A(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


# --- read phase 1, written phase 2: old value expected
@case("B_r1_w2_reshape", 4, lambda x: (x.view(2 * M, N // 2) + 1, x * 2))
def B(x):
    m, n = x.shape
    a = x.clone()
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    return out, a


@case("B2_r1_w2_nobarrier", 4, lambda x: (x.view(2 * M, N // 2) + 1, x * 2))
def B2(x):
    m, n = x.shape
    a = x.clone()
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    return out, a


# --- read and written in the same phase
@case("C1_same_loop_1loop", 2, lambda x: x * 2)
def C1(x):
    m, n = x.shape
    a = x.clone()
    v = a[:]
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = v[tm, tn] * 2.0
    return a


def _shift_ref(x):
    a = x.clone()
    a[: M - 8] = x[8:] * 2
    return a


@case("C4_same_tile_w_then_r_1loop", 3, lambda x: ((x * 2), (x * 2) + 1))
def C4(x):
    # one grid loop, the tile writes then reads through the view (eager: sees the write)
    m, n = x.shape
    a = torch.empty_like(x)
    v = a[:]
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            a[tm, tn] = x[tm, tn] * 2.0
        for tn in hl.tile(n):
            out[tm, tn] = v[tm, tn] + 1.0
    return a, out


@case("C5_w1_r2_and_w2", 4, lambda x: (x * 2 + 1,))
def C5(x):
    # phase 2 reads view and writes the base in place (read-after-write + same-phase)
    m, n = x.shape
    a = torch.empty_like(x)
    v = a[:]
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = v[tm, tn] + 1.0
    return (a,)


# --- two views of one tensor
@case(
    "D_two_views",
    6,
    lambda x: ((x * 2).view(2 * M, N // 2) + 1, (x * 2).view(M // 2, 2 * N) + 3),
)
def D(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    w = a.view(m // 2, n * 2)
    o1 = torch.empty_like(v)
    o2 = torch.empty_like(w)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        o1[tm, tn] = v[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m // 2, n * 2]):
        o2[tm, tn] = w[tm, tn] + 3.0
    return o1, o2


# --- view of an input, not written
@case("E1_input_view_reshape", 4, lambda x: (x * 2, x.view(2 * M, N // 2) + 1))
def E1(x):
    m, n = x.shape
    v = x.view(m * 2, n // 2)
    a = torch.empty_like(x)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return a, out


@case("E2_input_view_strided", 4, lambda x: (x * 2, x[:, ::2] + 1))
def E2(x):
    m, n = x.shape
    v = x[:, ::2]
    a = torch.empty_like(x)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return a, out


# --- an output being the view itself
@case("F1_two_written_views", 4, lambda x: (x * 2).view(2 * M, N // 2) + 1)
def F1(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        v[tm, tn] = v[tm, tn] + 1.0
    return v


@case(
    "F2_write_through_view_read_base",
    4,
    lambda x: ((x.view(2 * M, N // 2) * 2).view(M, N) + 1,),
)
def F2(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(x)
    xv = x.view(m * 2, n // 2)
    for tm, tn in hl.tile([m * 2, n // 2]):
        v[tm, tn] = xv[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = a[tm, tn] + 1.0
    return (out,)


# --- non-contiguous views
@case("G1_transpose_w1_r2", 4, lambda x: (x * 2)[:M, :M].t() + 1, None)
def G1(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a[:m, :m].t()
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, m]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


@case("G2_every_other_w1_r2", 4, lambda x: (x * 2)[:, ::2] + 1)
def G2(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a[:, ::2]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


@case("G3_every_other_r1_w2", 4, lambda x: (x[:, ::2] + 1, x * 2))
def G3(x):
    m, n = x.shape
    a = x.clone()
    v = a[:, ::2]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    return out, a


@case("G4_strided_inout_w1_r2", 4, lambda x: (x * 2)[:, :16:2] + 1)
def G4(x):
    # the WRITTEN tensor is strided: it is staged as a copy, the view must see it
    m, n = x.shape
    base = torch.empty_like(x)
    a = base[:, ::2]
    v = base[:, :16:2]
    out = torch.empty_like(v)
    y = x[:, ::2]
    for tm, tn in hl.tile([m, n // 2]):
        a[tm, tn] = y[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, 8]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


@case("G5_misaligned_view_w1_r2", 3, lambda x: (x * 2).view(-1)[1:] + 1)
def G5(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(-1)[1:]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for t in hl.tile(m * n - 1):
        out[t] = v[t] + 1.0
    return out


@case("G6_neg_view_w1_r2", 4, lambda x: -(x * 2) + 1)
def G6(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a._neg_view()
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


# --- interleaved strides sharing no elements
def _ilv_ref(x):
    return (x * 2, x + 1)


@case("H1_interleaved_cols_w1_r2", 4, _ilv_ref)
def H1(x):
    m, n = x.shape
    base = x.repeat_interleave(2, dim=1)
    a = base[:, ::2]
    v = base[:, 1::2]
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return a, out


@case("H2_interleaved_rows_w1_r2", 4, _ilv_ref)
def H2(x):
    m, n = x.shape
    base = x.repeat_interleave(2, dim=0)
    a = base[::2]
    v = base[1::2]
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return a, out


@case("H3_interleaved_written_contig_col", 4, lambda x: (x[:, 0:8] * 2, x[:, 8:16] + 1))
def H3(x):
    # submatrices side by side: spans overlap (row stride), no element shared
    m, n = x.shape
    base = x.clone()
    a = base[:, 0:8]
    v = base[:, 8:16]
    y = x[:, 0:8]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, 8]):
        a[tm, tn] = y[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, 8]):
        out[tm, tn] = v[tm, tn] + 1.0
    return a, out


# --- a view not overlapping the written tensor
@case("I1_disjoint_slices", 4, lambda x: (x * 2, x + 1))
def I1(x):
    m, n = x.shape
    base = x.repeat(2, 1)
    a = base[:m]
    v = base[m:]
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return a, out


# --- no barrier, several loops
@case("J1_nobarrier_w_then_r", 4, lambda x: (x * 2).view(2 * M, N // 2) + 1)
def J1(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


# --- the counter: phases that are sequential, not scf.forall
@case(
    "L1_seq_w1_forall_r2",
    4,
    lambda x: (x * 2).view(2 * M, N // 2) + 1,
    note="phase 1 is scf.for",
)
def L1(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, _tn in hl.tile([m, n]):
        a[tm, :] = x[tm, :] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


@case(
    "L2_forall_w1_seq_r2",
    4,
    lambda x: (x * 2).view(2 * M, N // 2) + 1,
    note="phase 2 is scf.for",
)
def L2(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, _tn in hl.tile([m * 2, n // 2]):
        out[tm, :] = v[tm, :] + 1.0
    return out


@case(
    "L3_seq_w1_seq_r2", 4, lambda x: (x * 2).view(2 * M, N // 2) + 1, note="0 foralls"
)
def L3(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, _tn in hl.tile([m, n]):
        a[tm, :] = x[tm, :] * 2.0
    hl.barrier()
    for tm, _tn in hl.tile([m * 2, n // 2]):
        out[tm, :] = v[tm, :] + 1.0
    return out


# --- declared parameters
def _p_args(x):
    return (torch.empty_like(x), x)


@case(
    "N1_param_reshape_alias_w1_r2",
    4,
    lambda a, x: (x * 2).view(2 * M, N // 2) + 1,
    args=_p_args,
)
def N1(a, x):
    m, n = x.shape
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


def _c_args(x):
    a = torch.empty_like(x)
    return (x, a, a.view(2 * M, N // 2))


@case(
    "N2_caller_view_w1_r2",
    4,
    lambda x, a, v: (x * 2).view(2 * M, N // 2) + 1,
    args=_c_args,
)
def N2(x, a, v):
    m, n = x.shape
    o = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        o[tm, tn] = v[tm, tn] + 1.0
    return o


def _inpl_args(x):
    z = x.clone()
    return (z, z, x)


@case(
    "N3_inplace_read_later_phase",
    4,
    lambda r, o, y: (y + 1, (y + 1) * 2),
    args=_inpl_args,
    note="r is o: eager phase 2 sees phase 1",
)
def N3(r, o, y):
    m, n = r.shape
    out = torch.empty_like(y)
    for tm, tn in hl.tile([m, n]):
        o[tm, tn] = y[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = r[tm, tn] * 2.0
    return o, out


@case(
    "N4_inplace_read_same_phase_only",
    4,
    lambda r, o, y: (y + 1, y * 3),
    args=_inpl_args,
    note="kernel(x, out=x), base correct",
)
def N4(r, o, y):
    # classic in-place: out[t] = r[t] + 1 where r is o; a second unrelated phase
    m, n = r.shape
    out = torch.empty_like(y)
    for tm, tn in hl.tile([m, n]):
        o[tm, tn] = r[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = y[tm, tn] * 3.0
    return o, out


# --- three phases: write, read, overwrite
@case("O_w1_r2_w3", 6, lambda x: ((x * 2).view(2 * M, N // 2) + 1, x * 5))
def O1(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 5.0
    return out, a


# --- single-loop controls
@case("R1_single_loop_view_of_unwritten", 2, lambda x: x.view(2 * M, N // 2) + 1)
def R1(x):
    m, n = x.shape
    a = x.clone()
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


@case("S_interleaved_single_loop", 2, lambda x: x * 2)
def S(x):
    m, n = x.shape
    base = x.repeat_interleave(2, dim=1)
    a = base[:, ::2]
    v = base[:, 1::2]
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = v[tm, tn] * 2.0
    return a


# ===== narrowed rule: tensor_param is None as the discriminator
@case(
    "N6_caller_view_r1_w2",
    4,
    lambda x, a, v: ((x).view(2 * M, N // 2) + 1, x * 2),
    args=_with_caller_view,
    note="caller view read before the write",
)
def N6(x, a, v):
    m, n = x.shape
    o = torch.empty_like(v)
    for tm, tn in hl.tile([m * 2, n // 2]):
        o[tm, tn] = v[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    return o, a


def _p1_ref(p, x):
    y = x.clone()
    y[:8] = x[:8] * 2
    return (y + 1,)


@case(
    "P1_host_slice_written_param_read_later",
    4,
    _p1_ref,
    args=lambda x: (x.clone(), x),
    note="written=host view (None), read=the param (has tensor_param)",
)
def P1(p, x):
    m, n = p.shape
    a = p[:8]
    y = x[:8]
    out = torch.empty_like(p)
    for tm, tn in hl.tile([8, n]):
        a[tm, tn] = y[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = p[tm, tn] + 1.0
    return (out,)


@case(
    "P2_host_reshape_written_param_read_later",
    4,
    lambda p, x: (x.view(2 * M, N // 2).view(M, N) * 2 + 1,),
    args=lambda x: (torch.empty_like(x), x),
    note="o = p.view is the written one, p is read",
)
def P2(p, x):
    m, n = p.shape
    o = p.view(m * 2, n // 2)
    xv = x.view(m * 2, n // 2)
    out = torch.empty_like(p)
    for tm, tn in hl.tile([m * 2, n // 2]):
        o[tm, tn] = xv[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = p[tm, tn] + 1.0
    return (out,)


def _rok(x):
    z = x.clone()
    return (z, z, x)


@case(
    "P3_host_slice_of_caller_alias",
    4,
    lambda r, o, y: ((y * 2)[:, ::2] + 1,),
    args=_rok,
    note="v = r[:, ::2], r is o from the caller",
)
def P3(r, o, y):
    m, n = r.shape
    v = r[:, ::2]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        o[tm, tn] = y[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return (out,)


@case(
    "P4_host_reshape_alias_of_caller_alias",
    4,
    lambda r, o, y: ((y * 2).view(2 * M, N // 2) + 1,),
    args=_rok,
    note="v = r.view is a same-element alias of param r; r is o from the caller",
)
def P4(r, o, y):
    m, n = r.shape
    v = r.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        o[tm, tn] = y[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return (out,)


@case("P5_view_of_view_local", 4, lambda x: (x * 2).view(M // 2, N * 2) + 1)
def P5(x):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    w = v.view(m // 2, n * 2)
    out = torch.empty_like(w)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m // 2, n * 2]):
        out[tm, tn] = w[tm, tn] + 1.0
    return out


@case(
    "P6_view_of_view_of_written_param",
    4,
    lambda p, x: ((x * 2)[:8, ::2] + 1,),
    args=lambda x: (torch.empty_like(x), x),
)
def P6(p, x):
    m, n = p.shape
    v = p[:8]
    w = v[:, ::2]
    out = torch.empty_like(w)
    for tm, tn in hl.tile([m, n]):
        p[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([8, n // 2]):
        out[tm, tn] = w[tm, tn] + 1.0
    return (out,)


@case("P7_rename_same_tensor", 4, lambda x: (x * 2 + 1,))
def P7(x):
    m, n = x.shape
    a = torch.empty_like(x)
    b = a
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = b[tm, tn] + 1.0
    return (out,)


G = torch.zeros(M, N)


def _g_args(x):
    G.copy_(x)
    return (G, x)


@case(
    "Q1_global_w1_r2",
    4,
    lambda p, x: (x * 2 + 1,),
    args=_g_args,
    note="module global G is the written param",
)
def Q1(p, x):
    m, n = p.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        p[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = G[tm, tn] + 1.0
    return (out,)


@case(
    "Q2_global_r1_w2",
    4,
    lambda p, x: (x + 1, x * 2),
    args=_g_args,
    note="global read before the write",
)
def Q2(p, x):
    m, n = p.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = G[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        p[tm, tn] = x[tm, tn] * 2.0
    return out, p


def _default_args(x):
    G.copy_(x)
    return (G, x)


@case(
    "Q4_default_arg_w1_r2",
    4,
    lambda p, x: (x * 2 + 1,),
    args=_default_args,
    note="tensor default argument",
)
def Q4(p, x, d=G):
    m, n = p.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        p[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = d[tm, tn] + 1.0
    return (out,)


@case(
    "Q5_global_view_written_by_host_local",
    4,
    lambda x: (x * 2 + 1,),
    args=lambda x: (G.copy_(x),) and (x,),
    note="a = G (global) written; v = G.view read",
)
def Q5(x):
    m, n = x.shape
    v = G.view(m * 2, n // 2).view(m, n)
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        G[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return (out,)


@case(
    "P8_param_slice_read_param_slice_written_disjoint_elems",
    4,
    lambda p, x: (x[:, 8:16] + 1, x[:, 0:8] * 2),
    args=lambda x: (x.clone(), x),
    note="host side-by-side submatrices of a param, no shared element",
)
def P8(p, x):
    m, n = p.shape
    a = p[:, 0:8]
    v = p[:, 8:16]
    y = x[:, 0:8]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, 8]):
        out[tm, tn] = v[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, 8]):
        a[tm, tn] = y[tm, tn] * 2.0
    return out, a


# phase 2 is a GEMM through the view
@case(
    "G7_gemm_through_view_w1_r2",
    6,
    lambda x, y: torch.matmul((x * 2).view(M * 2, N // 2), y),
    args=lambda x: (x, torch.ones(N // 2, 16) * 0.5),
)
def G7(x, y):
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty(m * 2, y.shape[1])
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, y.shape[1]]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(n // 2):
            acc = torch.addmm(acc, v[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


def _s2_ref(x):
    y = x.repeat(2, 1)
    r = y.clone()
    r[:24] = y[8:] * 2
    return r


@case("C2b_overlapping_shift_1loop", 2, _s2_ref, note="one loop; real overlap")
def C2b(x):
    n = x.shape[1]
    a = x.repeat(2, 1)
    w = a[:24]
    v = a[8:]
    for tm, tn in hl.tile([24, n]):
        w[tm, tn] = v[tm, tn] * 2.0
    return a


@case(
    "C3b_overlapping_shift_2phases",
    4,
    lambda x: (_s2_ref(x), x + 1),
    note="same-loop read+write in a 2-loop kernel; real overlap",
)
def C3b(x):
    m, n = x.shape
    a = x.repeat(2, 1)
    w = a[:24]
    v = a[8:]
    out = torch.empty_like(x)
    for tm, tn in hl.tile([24, n]):
        w[tm, tn] = v[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn] + 1.0
    return a, out


# --- the next iteration of a nested loop reads what this one stored
@case("U2_range_read_then_store", 2, lambda x: (x * 2 + 1,), note="dynamic range")
def U2(x):
    m, n = x.shape
    a = x.clone()
    v = a[:]
    out = torch.empty_like(x)
    c = 2
    for tm, tn in hl.tile([m, n]):
        for _ in range(c):
            out[tm, tn] = v[tm, tn] + 1.0
            a[tm, tn] = x[tm, tn] * 2.0
    return (out,)


@case(
    "U5_dynamic_while_read_then_store", 2, lambda x: (x * 2 + 1,), note="device while"
)
def U5(x):
    m, n = x.shape
    a = x.clone()
    v = a[:]
    out = torch.empty_like(x)
    c = torch.tensor(2, dtype=torch.int32)
    for tm, tn in hl.tile([m, n]):
        i = hl.zeros([], dtype=torch.int32)
        limit = hl.load(c, [])
        while i < limit:
            out[tm, tn] = v[tm, tn] + 1.0
            a[tm, tn] = x[tm, tn] * 2.0
            i = i + 1
    return (out,)


@case("U6_range_second_phase", 4, lambda x: (x * 3,), note="range loop in phase 2 only")
def U6(x):
    m, n = x.shape
    a = x.clone()
    v = a[:]
    out = torch.empty_like(x)
    c = 2
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn] * 0.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        for _ in range(c):
            out[tm, tn] = v[tm, tn]
            a[tm, tn] = x[tm, tn] * 3.0
    return (out,)


# --- right before the read order rule, refused now: conservative
def _t4(x):
    a = x.clone()
    a[:8] = x[:8] * 2
    return (a[8:] + 1,)


@case("T4_partial_store_elsewhere", 4, _t4, note="a[:8] stored, v = a[8:] read")
def T4(x):
    m, n = x.shape
    a = x.clone()
    v = a[8:]
    out = torch.empty_like(v)
    y = x[:8]
    for tm, tn in hl.tile([8, n]):
        a[tm, tn] = y[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m - 8, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return (out,)


@case(
    "T5_exclusive_if_branches",
    3,
    lambda x: (x + 1,),
    note="store and read: other branch",
)
def T5(x):
    m, n = x.shape
    a = x.clone()
    v = a[:]
    out = torch.empty_like(x)
    s = x.abs().max()
    for tm, tn in hl.tile([m, n]):
        if hl.load(s, []) > 1e6:
            a[tm, tn] = x[tm, tn] * 2.0
        else:
            out[tm, tn] = v[tm, tn] + 1.0
    return (out,)
