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