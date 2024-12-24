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

/*! \file snngrow/snngrow_backend/spikegemm/gemm/warp/spike_mma_simt.h
    *
    * Copyright (c) 2024 Beijing Institute of Technology AETAS Lab. and Utarn Technology Co., Ltd.  All rights reserved.
    *
    * Unless required by applicable law or agreed to in writing,
    * software distributed under the License is distributed on an
    * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
    * implied.
*/
#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"
#include "cutlass/functional.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
    /// Layout of A matrix
    typename LayoutA,
    /// Layout of B matrix
    typename LayoutB,
    /// Layout of C matrix
    typename LayoutC
>
struct Mma<
    gemm::GemmShape<1, 1, 1>, 1, bool, LayoutA, float, LayoutB, float, LayoutC, OpMultiplyAdd> {

    using Shape = gemm::GemmShape<1, 1, 1>;
    using Operator = OpMultiplyAdd;
    using ElementC = float;

    CUTLASS_HOST_DEVICE
    void operator()(
        Array<float, 1> &d,
        Array<bool, 1> const &a,
        Array<float, 1> const &b,
        Array<float, 1> const &c
    ) {
        if(a[0]) {
            d[0] = b[0] + c[0];
        }
        else {
            d[0] = c[0];
        }
    }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
    /// Layout of A matrix
    typename LayoutA,
    /// Layout of B matrix
    typename LayoutB,
    /// Layout of C matrix
    typename LayoutC
>
struct Mma<
    gemm::GemmShape<1, 1, 1>, 1, float, LayoutA, bool, LayoutB, float, LayoutC, OpMultiplyAdd> {

    using Shape = gemm::GemmShape<1, 1, 1>;
    using Operator = OpMultiplyAdd;
    using ElementC = float;

    CUTLASS_HOST_DEVICE
    void operator()(
        Array<float, 1> &d,
        Array<float, 1> const &a,
        Array<bool, 1> const &b,
        Array<float, 1> const &c
    ) {
        if(b[0]) {
            d[0] = a[0] + c[0];
        }
        else {
            d[0] = c[0];
        }
    }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
    /// Layout of A matrix
    typename LayoutA,
    /// Layout of B matrix
    typename LayoutB,
    /// Layout of C matrix
    typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, bool, LayoutA, double, LayoutB, double, LayoutC, OpMultiplyAdd> {

    using Shape = gemm::GemmShape<1, 1, 1>;
    using Operator = OpMultiplyAdd;
    using ElementC = double;

    CUTLASS_HOST_DEVICE
    void operator()(
        Array<double, 1> &d,
        Array<bool, 1> const &a,
        Array<double, 1> const &b,
        Array<double, 1> const &c
    ) {
        if(a[0]) {
            d[0] = b[0] + c[0];
        }
        else {
            d[0] = c[0];
        }
    }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
    /// Layout of A matrix
    typename LayoutA,
    /// Layout of B matrix
    typename LayoutB,
    /// Layout of C matrix
    typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, bool, LayoutA, int, LayoutB, int, LayoutC, OpMultiplyAdd> {

    using Shape = gemm::GemmShape<1, 1, 1>;
    using Operator = OpMultiplyAdd;
    using ElementC = int;

    CUTLASS_HOST_DEVICE
    void operator()(
        Array<int, 1> &d,
        Array<bool, 1> const &a,
        Array<int, 1> const &b,
        Array<int, 1> const &c
    ) {
        if(a[0]) {
            d[0] = b[0] + c[0];
        }
        else {
            d[0] = c[0];
        }
    }
};

} // namespace arch
} // namespace cutlass