/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file snngrow/snngrow_backend/spikegemm/gemm/threadblock/spike_mma_core.h
    *
    * Copyright (c) 2024 Beijing Institute of Technology AETAS Lab. and Utarn Technology Co., Ltd.  All rights reserved.
    *
    * Unless required by applicable law or agreed to in writing,
    * software distributed under the License is distributed on an
    * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
    * implied.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/mma_singlestage.h"
#include "cutlass/arch/cache_operation.h" 
#include "cutlass/arch/mma.h"

#include "gemm/warp/spike_mma_simt_policy.h"
#include "gemm/warp/spike_mma_simt.h"

namespace cutlass {
namespace gemm {
namespace threadblock {
    /////////////////////////////////////////////////////////////////////////////////////////////////

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Number of stages
    int Stages = 2,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           cutlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global,
    /// per-element transformation for elements of A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// per-element transformation for elements of B
    ComplexTransform TransformB = ComplexTransform::kNone,
    bool IsComplex = false // (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct SpikeMmaCore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct SpikeMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, bool,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = bool;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    /// Default Operator
    using Operator = Operator_;

    /// Number of warps present
    using WarpCount = GemmShape<
        Shape::kM / WarpShape::kM,
        Shape::kN / WarpShape::kN,
        PartitionsK
    >;

    // Divisility requirements
    static_assert(
        !(Shape::kM % WarpShape::kM) &&
        !(Shape::kN % WarpShape::kN),
        "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
    );

    /// Number of threads per warp
    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kElementsPerAccess = 1;
    
    //
    // Shared memory layouts
    //

    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator A
    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kK, Shape::kM>,
        kThreads,
        kElementsPerAccess
    >;

    /// Transpose the ThreadMap of iterator A
    using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    /// Shared memory iterator to A operand
    using SmemIteratorA = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, 
        ElementA, 
        SmemLayoutA,
        1,
        SmemThreadMapA
    >;

    /// Policy of iterator B
    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kN, Shape::kK>,
        kThreads,
        kElementsPerAccess
    >;

    /// Shared memory iterator to B operand
    using SmemIteratorB = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kK, Shape::kN>, 
        ElementB, 
        SmemLayoutB,
        0,
        IteratorThreadMapB
    >;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level op
    static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
        "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

    static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);

    static_assert(!(kPaddingM % LaneM),
                    "Padding must be divisible by Lane");

    // these should have max of thread tile also
    using LaneMmaShape = cutlass::gemm::GemmShape<
        LaneM,
        LaneN,
        1>;
    using Policy = cutlass::gemm::warp::SpikeMmaSimtPolicy<
        cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
        cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
        LaneMmaShape
    >;

    using MmaWarpSimt = cutlass::gemm::warp::SpikeMmaSimt<
        WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
        ElementA,     /// Data type of A elements
        SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
        ElementB,     /// Data type of B elements
        SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
        ElementC,     /// Element type of C matrix
        LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
        Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    >;

    /// Policy used to define MmaPipelined 
    using MmaPolicy = MmaPolicy<
        MmaWarpSimt,
        MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
        MatrixShape<0, 0>,
        WarpCount::kK
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of B operand
    typename ElementA_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct SpikeMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::RowMajor, bool, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = bool;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    /// Default Operator
    using Operator = Operator_;

    /// Number of warps present
    using WarpCount = GemmShape<
        Shape::kM / WarpShape::kM,
        Shape::kN / WarpShape::kN,
        PartitionsK
    >;

    // Divisility requirements
    static_assert(
        !(Shape::kM % WarpShape::kM) &&
        !(Shape::kN % WarpShape::kN),
        "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
    );

    /// Number of threads per warp
    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kElementsPerAccess = 1;
    
    //
    // Shared memory layouts
    //

    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator A
    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kK, Shape::kM>,
        kThreads,
        kElementsPerAccess
    >;

    /// Transpose the ThreadMap of iterator A
    using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    /// Shared memory iterator to A operand
    using SmemIteratorA = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, 
        ElementA, 
        SmemLayoutA,
        1,
        SmemThreadMapA
    >;

    /// Policy of iterator B
    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kN, Shape::kK>,
        kThreads,
        kElementsPerAccess
    >;

    /// Shared memory iterator to B operand
    using SmemIteratorB = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kK, Shape::kN>, 
        ElementB, 
        SmemLayoutB,
        0,
        IteratorThreadMapB
    >;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level op
    static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
        "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

    static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);

    static_assert(!(kPaddingM % LaneM),
                    "Padding must be divisible by Lane");

    // these should have max of thread tile also
    using LaneMmaShape = cutlass::gemm::GemmShape<
        LaneM,
        LaneN,
        1>;
    using Policy = cutlass::gemm::warp::SpikeMmaSimtPolicy<
        cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
        cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
        LaneMmaShape
    >;

    using MmaWarpSimt = cutlass::gemm::warp::SpikeMmaSimt<
        WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
        ElementA,     /// Data type of A elements
        SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
        ElementB,     /// Data type of B elements
        SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
        ElementC,     /// Element type of C matrix
        LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
        Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    >;

    /// Policy used to define MmaPipelined 
    using MmaPolicy = MmaPolicy<
        MmaWarpSimt,
        MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
        MatrixShape<0, 0>,
        WarpCount::kK
    >;
};
} // namespace threadblock
} // namespace gemm 
} // namespace cutlass
