# SPDX-License-Identifier: Apache-2.0
"""Prefill word-scale loads with the existing Q0 FP32 scatter contract.

Only the two producer loads change. Q0 route publication, MMA, contribution
rounding and FP32 scatter remain inherited from the pinned Q0 implementation.
This candidate is admitted only for 65..8192 rows, never a decode batch.
"""
from .moe_dynamic_gated_sf6_q0 import MoEGatedDynamicKernelSF6Q0
from .moe_dynamic_gated_sf6_words import MoEGatedDynamicKernelSF6Words, stock_contract_matches


class MoEGatedDynamicKernelSF6Q0Words(MoEGatedDynamicKernelSF6Q0):
    def __init__(self, *args, **kwargs):
        if not stock_contract_matches():
            raise RuntimeError("prefill Q0 word decoder parent source drifted")
        super().__init__(*args, **kwargs)
        if not self.reform_sf_pack:
            raise ValueError("prefill Q0 word decoder requires prepared SF6 scales")
        self.prefill_word_unpack = True

    load_fc1_tma_slice = MoEGatedDynamicKernelSF6Words.load_fc1_tma_slice
    load_fc2_tma_tile = MoEGatedDynamicKernelSF6Words.load_fc2_tma_tile
