#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "cutlass/layout/permute.h"

#include "gemm/threadblock/spike_mma.h"
#include "gemm/kernel/gemm_spike_kernel.h"

namespace cutlass {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute,
    ///
    typename Enable = void
>
struct SpikeGemm;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages
    int Stages,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct SpikeGemm<ElementA,
                   LayoutA,
                   kAlignmentA,
                   ElementB,
                   LayoutB,
                   kAlignmentB,
                   ElementC,
                   LayoutC,
                   ElementAccumulator,
                   arch::OpClassSimt,
                   ArchTag,
                   ThreadblockShape,
                   WarpShape,
                   GemmShape<1, 1, 1>,
                   EpilogueOutputOp,
                   ThreadblockSwizzle,
                   Stages,
                   SplitKSerial,
                   Operator,
                   SharedMemoryClear,
                   GatherA,
                   GatherB,
                   ScatterD,
                   PermuteDLayout,
                   PermuteALayout,
                   PermuteBLayout> {
    static_assert((platform::is_same<LayoutC, layout::RowMajor>::value
            || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
            "Epilogue in the kernel level must be row major");
    
    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::gemm::threadblock::SpikeMma<
        ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
        ElementAccumulator, LayoutC, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, Stages,
        Operator, false, SharedMemoryClear, GatherA, GatherB,
        PermuteALayout, PermuteBLayout>::ThreadblockMma;
    
    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

    /// Define the epilogue
    using RegularEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        ThreadblockShape,
        typename Mma::Operator,
        EpilogueOutputOp,
        kEpilogueElementsPerAccess,
        ScatterD,
        PermuteDLayout
        >::Epilogue;

    using Affine2Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimtAffineRankN<
        2,
        ThreadblockShape,
        typename Mma::Operator,
        EpilogueOutputOp,
        kEpilogueElementsPerAccess
        >::Epilogue;

    using Epilogue = typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value,
                                                    RegularEpilogue,
                                                    Affine2Epilogue>::type;

    /// Define the kernel-level GEMM operator.
    using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>; 

};
}
}
}