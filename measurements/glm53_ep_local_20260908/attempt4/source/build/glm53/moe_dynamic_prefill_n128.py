"""Opt-in paired-N128 FC1 for the exact GLM dynamic-prefill lane.

The N128 gate/up pair visits each FC1 K tile once, instead of the inherited
N64 pair's two visits. Per unsplit M128/I512 task, A+SFA requests fall from
2.25 to 1.125 MiB and W13 SFB from 0.5 to 0.25 MiB. All TMA payload falls
from 5.875 to 4.5 MiB (23.4%). These are requested bytes, not measured DRAM
traffic or a latency prediction: cache hits, two-stage stalls and register
pressure can offset the reduction.

No weight repack or shared allocation is added. Gate B has two stages in
sB (16 KiB); Up B has two stages in startup/epilogue-idle sC (16 KiB).
A5, SFA4, Q1 and FC2 retain their original K128 layouts. The producer waits
for prior Q1 before next-slice Stage0 because both B stages now alias sC.
Every output keeps its increasing-K64 MMA accumulation order, gate/up
activation and BF16 conversion. Task routing and atomic scatter remain the
parent reuse candidate's contract. This lane also includes its Q0/FC2 reuse.

VLLM_GLM53_B12X_PREFILL_FC1_N128=1 is required at dispatch. Defaults, small
batches and decode remain stock. Compile/spill, GPU numerical and throughput
validation have deliberately not been run for this implementation.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

from ._moe_dynamic.gated import _dynamic_gated_activation_f32
from .moe_dynamic_prefill import MoEGatedPrefillReuseKernel


class MoEGatedPrefillN128Kernel(MoEGatedPrefillReuseKernel):
    """One N128 gate/up pair per intermediate slice, with two FC1 stages."""

    prefill_fc1_n128 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1_tile_shape_mnk = self.tile_shape_mnk
        self.fc1_sfb_tile_shape_nk = (128, 128)
        self.fc1_sfb_tiles_per_block = 1
        # One CTA is already resident per SM. Keep the producer's 32-register
        # reservation and allow 248 for each of eight math warps (<64K total).
        self.mma_register_requirement = 248

    def _setup_attributes(self, hidden_size: int):
        super()._setup_attributes(hidden_size)
        # The parent builds the N128 MMA/permutation from the shape above.
        # Preserve its A5/B2/SFA4/FC2 Stage3 storage, replacing only FC1's
        # descriptor/barrier geometry. Both FC1 B branches need two stages.
        self.ab_stage = 2
        self.ab_storage_stage = 2
        (
            _a, self.fc1_b_smem_layout_staged, _sfa,
            self.fc1_sfb_smem_layout_staged, _epi,
        ) = self._dense_cls._make_smem_layouts(
            self.fc1_tile_shape_mnk, self.epi_tile,
            self.a_dtype, self.a_layout, self.b_dtype, self.b_layout,
            self.ab_stage, cutlass.BFloat16, self.c_layout, self.epi_stage,
            self.sf_vec_size, self.fc1_tiled_mma,
        )
        self.fc1_b_smem_layout_storage = self.fc1_b_smem_layout_staged
        self.fc1_sfb_smem_layout_storage = self.fc1_sfb_smem_layout_staged

    @cute.jit
    def fc1_gate_up_swiglu_to_sC(
        self,
        tidx,
        ml_pipeline,
        cons_state,
        up_pipeline,
        up_cons_state,
        fc1_tiled_mma,
        mma_atom,
        sSFB_fc1,
        sSFB_up_fc1,
        sSFB_up_fc1_extra,
        fc1_thr_mma,
        thr_ld_SFB_fc1,
        csA_fc1,
        csB_fc1,
        csB_up_fc1,
        csSFA_fc1,
        crA_fc1,
        crB_fc1,
        crB_up_fc1,
        crSFA_fc1,
        tCrA_fc1,
        tCrB_fc1,
        tCrB_up_fc1,
        tCrSFA_fc1,
        smem_copy_A_fc1,
        smem_copy_B_fc1,
        smem_copy_SFA_fc1,
        smem_copy_SFB_fc1,
        fc1_tiled_copy_r2s,
        fc1_tRS_sD,
        fc1_k_tile_cnt,
        fc1_num_k_blocks,
        fc1_m_tiles,
        fc1_n_tiles,
        alpha_value,
        valid_rows,
        task_expert_idx,
        global_scale,
        sC,
        sA,
        sfa_base_addr,
        epi_rest_m,
    ):
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        fc1_acc_shape = fc1_tiled_mma.partition_shape_C(
            (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1])
        )
        gate_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        fc1_tRS_rGate = fc1_tiled_copy_r2s.retile(gate_acc)
        fc1_tRS_rUp = fc1_tiled_copy_r2s.retile(up_acc)
        fc1_tRS_rAct = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, self.acc_dtype
        )
        fc1_tRS_rAct_out = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, cutlass.BFloat16
        )

        # ============================================================
        # PHASE A: native paired-N128 FC1 for this logical N128 slice
        # ============================================================
        # One pipeline/state sequence covers the complete N128 pair.
        # Each stage carries one A/SFA plus independent
        # Gate/Up B/SFB payloads; both OMMAs complete before release.
        cons_state.reset_count()
        for fc1_half in cutlass.range_constexpr(1):
            # SM120 packs SFB in physical N128 blocks.  Select the
            # complete N128 scale block for both branches (one visit).
            sSFB_fc1_half = cute.local_tile(
                sSFB_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_fc1_half)
            crSFB_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_fc1_half)
            sSFB_up_fc1_half = cute.local_tile(
                sSFB_up_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_up_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_up_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_up_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_half)
            sSFB_up_fc1_extra_half = cute.local_tile(
                sSFB_up_fc1_extra,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0),
            )
            csSFB_up_fc1_extra_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_extra_half)
            crSFB_up_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_up_fc1_half)
            fz_crSFA_fc1 = cute.filter_zeros(crSFA_fc1)
            fz_crSFB_fc1_half = cute.filter_zeros(crSFB_fc1_half)
            fz_crSFB_up_fc1_half = cute.filter_zeros(crSFB_up_fc1_half)

            # Branch-paired Gate/Up N128: A/SFA are read once from
            # this pipeline stage and feed both OMMAs.
            gate_acc.fill(0.0)
            up_acc.fill(0.0)
            peek = ml_pipeline.consumer_try_wait(cons_state)
            ml_pipeline.consumer_wait(cons_state, peek)
            csA_p = csA_fc1[None, None, None, cons_state.index]
            csB_p = csB_fc1[None, None, None, cons_state.index]
            csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
            csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
            csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
            csSFB_up_p = csSFB_up_fc1_half[None, None, None, Int32(0)]
            if cons_state.index < Int32(self.ab_storage_stage):
                csSFB_up_p = csSFB_up_fc1_half[None, None, None, cons_state.index]
            else:
                csSFB_up_p = csSFB_up_fc1_extra_half
            cute.copy(
                smem_copy_A_fc1,
                csA_p[None, None, 0],
                crA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_p[None, None, 0],
                crB_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_up_p[None, None, 0],
                crB_up_fc1[None, None, 0],
            )
            fz_csSFA_p = cute.filter_zeros(csSFA_p)
            fz_csSFB_p = cute.filter_zeros(csSFB_p)
            fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
            cute.copy(
                smem_copy_SFA_fc1,
                fz_csSFA_p[None, None, 0],
                fz_crSFA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_p[None, None, 0],
                fz_crSFB_fc1_half[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_up_p[None, None, 0],
                fz_crSFB_up_fc1_half[None, None, 0],
            )
            for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                    k_next = (
                        0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == fc1_num_k_blocks - 1:
                        ml_pipeline.consumer_release(cons_state)
                        cons_state.advance()
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        csA_p = csA_fc1[None, None, None, cons_state.index]
                        csB_p = csB_fc1[None, None, None, cons_state.index]
                        csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
                        csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
                        csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_p = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_p = csSFB_up_fc1_extra_half
                        fz_csSFA_p = cute.filter_zeros(csSFA_p)
                        fz_csSFB_p = cute.filter_zeros(csSFB_p)
                        fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
                        ml_pipeline.consumer_wait(cons_state, peek)
                    # Issue current Gate MMA first so the following
                    # LDS can overlap tensor-pipe work.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                gate_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_fc1[None, _nt, k_block_idx],
                                gate_acc[None, _mt, _nt],
                            )
                    if k_next > 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
                    # Current Up consumes only current fragments;
                    # next-K64 fragments remain independent.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                up_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_up_fc1[None, _nt, k_block_idx],
                                up_acc[None, _mt, _nt],
                            )
                    # Preserve V98's conservative cross-stage
                    # boundary: load next-stage K64(0) after Up.
                    if k_next == 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
            for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                k_next = 0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                if k_block_idx == fc1_num_k_blocks - 1:
                    ml_pipeline.consumer_release(cons_state)
                    cons_state.advance()
                if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                    cute.copy(
                        smem_copy_A_fc1,
                        csA_p[None, None, k_next],
                        crA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_p[None, None, k_next],
                        crB_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_up_p[None, None, k_next],
                        crB_up_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFA_fc1,
                        fz_csSFA_p[None, None, k_next],
                        fz_crSFA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_p[None, None, k_next],
                        fz_crSFB_fc1_half[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_up_p[None, None, k_next],
                        fz_crSFB_up_fc1_half[None, None, k_next],
                    )
                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                    for _nt in cutlass.range_constexpr(fc1_n_tiles):
                        mma_atom.set(
                            WarpField.SFA,
                            tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                        )
                        cute.gemm(
                            mma_atom,
                            gate_acc[None, _mt, _nt],
                            tCrA_fc1[None, _mt, k_block_idx],
                            tCrB_fc1[None, _nt, k_block_idx],
                            gate_acc[None, _mt, _nt],
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                        )
                        cute.gemm(
                            mma_atom,
                            up_acc[None, _mt, _nt],
                            tCrA_fc1[None, _mt, k_block_idx],
                            tCrB_up_fc1[None, _nt, k_block_idx],
                            up_acc[None, _mt, _nt],
                        )

            # Up's two B stages occupy the first half of sC. All math
            # warps must finish reading both stages before activation stores
            # reclaim that region. The producer waits for Q1 before the next
            # slice's very first stage; no Stage0/1 prefetch aliases live sC.
            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilog_sync_barrier.arrive_and_wait()

            # Materialize the full N128 activation into the unchanged Q1
            # layout; each element retains the original K64 accumulation order.
            for mma_m in cutlass.range_constexpr(fc1_m_tiles):
                for mma_n in cutlass.range_constexpr(fc1_n_tiles):
                    full_mma_n = fc1_half * fc1_n_tiles + mma_n
                    gate_slice = fc1_tRS_rGate[(None, mma_m, mma_n)]
                    up_slice = fc1_tRS_rUp[(None, mma_m, mma_n)]
                    for elem_idx in cutlass.range_constexpr(cute.size(fc1_tRS_rAct)):
                        g = alpha_value * gate_slice[elem_idx]
                        u = alpha_value * up_slice[elem_idx]
                        fc1_tRS_rAct[elem_idx] = _dynamic_gated_activation_f32(
                            g,
                            u,
                            activation=self.activation,
                            limit=self.swiglu_limit,
                            alpha=self.swiglu_alpha,
                            beta=self.swiglu_beta,
                            fast_math=self.fast_math,
                        )
                    act_vec = fc1_tRS_rAct.load()
                    act_vec = act_vec.to(cutlass.BFloat16)
                    fc1_tRS_rAct_out.store(act_vec)
                    cute.copy(
                        fc1_tiled_copy_r2s,
                        fc1_tRS_rAct_out,
                        fc1_tRS_sD[(None, mma_m, full_mma_n, 0)],
                    )

        return cons_state, up_cons_state

    @cute.jit
    def fc1_gate_up_swiglu_to_sC_tail(
        self,
        tidx,
        ml_pipeline,
        cons_state,
        up_pipeline,
        up_cons_state,
        fc1_tiled_mma,
        mma_atom,
        sSFB_fc1,
        sSFB_up_fc1,
        sSFB_up_fc1_extra,
        fc1_thr_mma,
        thr_ld_SFB_fc1,
        csA_fc1,
        csB_fc1,
        csB_up_fc1,
        csSFA_fc1,
        crA_fc1,
        crB_fc1,
        crB_up_fc1,
        crSFA_fc1,
        tCrA_fc1,
        tCrB_fc1,
        tCrB_up_fc1,
        tCrSFA_fc1,
        smem_copy_A_fc1,
        smem_copy_B_fc1,
        smem_copy_SFA_fc1,
        smem_copy_SFB_fc1,
        fc1_tiled_copy_r2s,
        fc1_tRS_sD,
        fc1_k_tile_cnt,
        fc1_num_k_blocks,
        fc1_m_tiles,
        fc1_n_tiles,
        alpha_value,
        valid_rows,
        warp_m_coord: Int32,
        task_expert_idx,
        global_scale,
        sC,
        sA,
        sfa_base_addr,
        epi_rest_m,
    ):
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        fc1_acc_shape = fc1_tiled_mma.partition_shape_C(
            (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1])
        )
        gate_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        fc1_tRS_rGate = fc1_tiled_copy_r2s.retile(gate_acc)
        fc1_tRS_rUp = fc1_tiled_copy_r2s.retile(up_acc)
        fc1_tRS_rAct = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, self.acc_dtype
        )
        fc1_tRS_rAct_out = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, cutlass.BFloat16
        )

        # ============================================================
        # PHASE A: native paired-N128 FC1 for this logical N128 slice
        # ============================================================
        # One pipeline/state sequence covers the complete N128 pair.
        # Each stage carries one A/SFA plus independent
        # Gate/Up B/SFB payloads; both OMMAs complete before release.
        cons_state.reset_count()
        for fc1_half in cutlass.range_constexpr(1):
            # SM120 packs SFB in physical N128 blocks.  Select the
            # complete N128 scale block for both branches (one visit).
            sSFB_fc1_half = cute.local_tile(
                sSFB_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_fc1_half)
            crSFB_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_fc1_half)
            sSFB_up_fc1_half = cute.local_tile(
                sSFB_up_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_up_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_up_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_up_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_half)
            sSFB_up_fc1_extra_half = cute.local_tile(
                sSFB_up_fc1_extra,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0),
            )
            csSFB_up_fc1_extra_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_extra_half)
            crSFB_up_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_up_fc1_half)
            fz_crSFA_fc1 = cute.filter_zeros(crSFA_fc1)
            fz_crSFB_fc1_half = cute.filter_zeros(crSFB_fc1_half)
            fz_crSFB_up_fc1_half = cute.filter_zeros(crSFB_up_fc1_half)

            # Branch-paired Gate/Up N128: A/SFA are read once from
            # this pipeline stage and feed both OMMAs.
            gate_acc.fill(0.0)
            up_acc.fill(0.0)
            peek = ml_pipeline.consumer_try_wait(cons_state)
            ml_pipeline.consumer_wait(cons_state, peek)
            csA_p = csA_fc1[None, None, None, cons_state.index]
            csB_p = csB_fc1[None, None, None, cons_state.index]
            csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
            csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
            csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
            csSFB_up_p = csSFB_up_fc1_half[None, None, None, Int32(0)]
            if cons_state.index < Int32(self.ab_storage_stage):
                csSFB_up_p = csSFB_up_fc1_half[None, None, None, cons_state.index]
            else:
                csSFB_up_p = csSFB_up_fc1_extra_half
            cute.copy(
                smem_copy_A_fc1,
                csA_p[None, None, 0],
                crA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_p[None, None, 0],
                crB_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_up_p[None, None, 0],
                crB_up_fc1[None, None, 0],
            )
            fz_csSFA_p = cute.filter_zeros(csSFA_p)
            fz_csSFB_p = cute.filter_zeros(csSFB_p)
            fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
            cute.copy(
                smem_copy_SFA_fc1,
                fz_csSFA_p[None, None, 0],
                fz_crSFA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_p[None, None, 0],
                fz_crSFB_fc1_half[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_up_p[None, None, 0],
                fz_crSFB_up_fc1_half[None, None, 0],
            )
            for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                    k_next = (
                        0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == fc1_num_k_blocks - 1:
                        ml_pipeline.consumer_release(cons_state)
                        cons_state.advance()
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        csA_p = csA_fc1[None, None, None, cons_state.index]
                        csB_p = csB_fc1[None, None, None, cons_state.index]
                        csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
                        csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
                        csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_p = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_p = csSFB_up_fc1_extra_half
                        fz_csSFA_p = cute.filter_zeros(csSFA_p)
                        fz_csSFB_p = cute.filter_zeros(csSFB_p)
                        fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
                        ml_pipeline.consumer_wait(cons_state, peek)
                    # Issue current Gate MMA first so the following
                    # LDS can overlap tensor-pipe work.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                            for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    gate_acc[None, _mt, _nt],
                                    tCrA_fc1[None, _mt, k_block_idx],
                                    tCrB_fc1[None, _nt, k_block_idx],
                                    gate_acc[None, _mt, _nt],
                                )
                    if k_next > 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
                    # Current Up consumes only current fragments;
                    # next-K64 fragments remain independent.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                            for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    up_acc[None, _mt, _nt],
                                    tCrA_fc1[None, _mt, k_block_idx],
                                    tCrB_up_fc1[None, _nt, k_block_idx],
                                    up_acc[None, _mt, _nt],
                                )
                    # Preserve V98's conservative cross-stage
                    # boundary: load next-stage K64(0) after Up.
                    if k_next == 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
            for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                k_next = 0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                if k_block_idx == fc1_num_k_blocks - 1:
                    ml_pipeline.consumer_release(cons_state)
                    cons_state.advance()
                if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                    cute.copy(
                        smem_copy_A_fc1,
                        csA_p[None, None, k_next],
                        crA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_p[None, None, k_next],
                        crB_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_up_p[None, None, k_next],
                        crB_up_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFA_fc1,
                        fz_csSFA_p[None, None, k_next],
                        fz_crSFA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_p[None, None, k_next],
                        fz_crSFB_fc1_half[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_up_p[None, None, k_next],
                        fz_crSFB_up_fc1_half[None, None, k_next],
                    )
                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                    if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                gate_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_fc1[None, _nt, k_block_idx],
                                gate_acc[None, _mt, _nt],
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                up_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_up_fc1[None, _nt, k_block_idx],
                                up_acc[None, _mt, _nt],
                            )

            # Up's two B stages occupy the first half of sC. All math
            # warps must finish reading both stages before activation stores
            # reclaim that region. The producer waits for Q1 before the next
            # slice's very first stage; no Stage0/1 prefetch aliases live sC.
            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilog_sync_barrier.arrive_and_wait()

            # Materialize the full N128 activation into the unchanged Q1
            # layout; each element retains the original K64 accumulation order.
            for mma_m in cutlass.range_constexpr(fc1_m_tiles):
                if valid_rows > Int32(mma_m * 64) + warp_m_coord * Int32(16):
                    for mma_n in cutlass.range_constexpr(fc1_n_tiles):
                        full_mma_n = fc1_half * fc1_n_tiles + mma_n
                        gate_slice = fc1_tRS_rGate[(None, mma_m, mma_n)]
                        up_slice = fc1_tRS_rUp[(None, mma_m, mma_n)]
                        for elem_idx in cutlass.range_constexpr(
                            cute.size(fc1_tRS_rAct)
                        ):
                            g = alpha_value * gate_slice[elem_idx]
                            u = alpha_value * up_slice[elem_idx]
                            fc1_tRS_rAct[elem_idx] = _dynamic_gated_activation_f32(
                                g,
                                u,
                                activation=self.activation,
                                limit=self.swiglu_limit,
                                alpha=self.swiglu_alpha,
                                beta=self.swiglu_beta,
                                fast_math=self.fast_math,
                            )
                        act_vec = fc1_tRS_rAct.load()
                        act_vec = act_vec.to(cutlass.BFloat16)
                        fc1_tRS_rAct_out.store(act_vec)
                        cute.copy(
                            fc1_tiled_copy_r2s,
                            fc1_tRS_rAct_out,
                            fc1_tRS_sD[(None, mma_m, full_mma_n, 0)],
                        )

        return cons_state, up_cons_state

    @cute.jit
    def load_fc1_tma_slice(
        self,
        intermediate_slice: Int32,
        wait_for_prior_slice: Int32,
        task_expert_idx: Int32,
        gate_tile_cnt,
        fc1_k_tile_cnt,
        prod_state,
        ml_pipeline,
        up_prod_state,
        up_pipeline,
        tma_inputs,
        gmem_partitions,
        smem_partitions,
    ):
        tma_a, tma_b_w13, tma_sfa, tma_sfb_w13 = tma_inputs
        tAgA_mk, tAgSFA_mk, tBgB_w13, tBgSFB_w13 = gmem_partitions
        (
            tAsA,
            tAsSFA,
            tBsB_w13,
            tBsB_w13_up,
            tBsSFB_w13,
            tBsSFB_w13_up,
            tBsSFB_w13_up_extra,
        ) = smem_partitions

        # FC1 producer follows the same continuous order as the
        # consumer.  Each logical N128 slice maps to one native B128
        # pair. Gate/Up share one A/SFA stage and
        # use independent B/SFB destinations under one barrier.
        prod_state.reset_count()
        gate_wait_pending = wait_for_prior_slice
        for fc1_half in cutlass.range_constexpr(1):
            native_up_slice_idx = intermediate_slice + Int32(fc1_half)
            native_gate_slice_idx = intermediate_slice + gate_tile_cnt + Int32(fc1_half)
            tBgB_w13_gate_nk = tBgB_w13[
                (
                    None,
                    native_gate_slice_idx,
                    None,
                    task_expert_idx,
                )
            ]
            tBgB_w13_up_nk = tBgB_w13[
                (
                    None,
                    native_up_slice_idx,
                    None,
                    task_expert_idx,
                )
            ]
            tBgSFB_w13_gate_nk = tBgSFB_w13[
                (
                    None,
                    intermediate_slice + gate_tile_cnt,
                    None,
                    task_expert_idx,
                )
            ]
            tBgSFB_w13_up_nk = tBgSFB_w13[
                (
                    None,
                    intermediate_slice,
                    None,
                    task_expert_idx,
                )
            ]

            # ---- Branch-paired Gate/Up N128 ----
            for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                # Up uses sC starting at Stage0. Wait before either of
                # the next slice's stages can overwrite its prior Q1 input.
                if gate_wait_pending > Int32(0):
                    self.pass_gate_barrier.wait_unaligned()
                    gate_wait_pending = Int32(0)
                ml_pipeline.producer_acquire(prod_state)
                cute.copy(
                    tma_a,
                    tAgA_mk[(None, k_tile)],
                    tAsA[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_b_w13,
                    tBgB_w13_gate_nk[(None, k_tile)],
                    tBsB_w13[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_b_w13,
                    tBgB_w13_up_nk[(None, k_tile)],
                    tBsB_w13_up[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_sfa,
                    tAgSFA_mk[(None, k_tile)],
                    tAsSFA[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_sfb_w13,
                    tBgSFB_w13_gate_nk[(None, k_tile)],
                    tBsSFB_w13[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                if prod_state.index < Int32(self.ab_storage_stage):
                    cute.copy(
                        tma_sfb_w13,
                        tBgSFB_w13_up_nk[(None, k_tile)],
                        tBsSFB_w13_up[(None, prod_state.index)],
                        tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                    )
                else:
                    cute.copy(
                        tma_sfb_w13,
                        tBgSFB_w13_up_nk[(None, k_tile)],
                        tBsSFB_w13_up_extra,
                        tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                    )
                ml_pipeline.producer_commit(prod_state)
                prod_state.advance()

        return prod_state, up_prod_state


__all__ = ["MoEGatedPrefillN128Kernel"]
