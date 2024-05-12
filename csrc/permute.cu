/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <stdexcept>

#include "permute.h"

#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using torch::Tensor;

namespace grouped_gemm {

template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Top K
//
/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void moe_permute_topK_row_map(const int *sorted_row_id,
                                         int *row_id_map,
                                         const int num_rows,
                                         const int num_topK)
{
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    row_id_map[source_topK_id * num_rows + source_token_id] = idx;
}

__global__ void moe_permute_topK_row_map_v2(const int *sorted_row_id,
                                            int *row_id_map,
                                            const int num_rows,
                                            const int num_topK,
                                            bool inverted = false /* row_id_map is dest row id <source_topK_id, source_token_id> of routed tokens to source tokens row id position map if inverted is false*/)
{
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]

    // We remap data on chip to coalesce access to global memory when store data by inverting key val pairs
    cg::thread_block cta = cg::this_thread_block();

    // Staticly allocate s_mem 16 from internal memory pool
    extern __shared__ int8_t s_mem[];

    // const int BLOCK_SIZE_ROWS = 1;

    int32_t *tile = reinterpret_cast<int32_t *>(s_mem);
    int32_t *remapped_key_slot = tile;
    // int32_t *remapped_val_slot = tile + BLOCK_SIZE_ROWS * num_topK;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int in_idx = bid * blockDim.x + tid;
    const int& out_idx =  in_idx;

    if (in_idx >= num_rows * num_topK)
        return;
    
    // Load : load a row (by half a warp) from
    const int& source_row = sorted_row_id[in_idx];
    
    // Compute : remap data tile; the randomly access to share mem is bank conflicts free
    const int source_token_id = source_row / num_topK;
    const int source_topK_id = source_row % num_topK;
    
    // NOTE(yiakwy) : reorder source rows (of sorted_row_id which sorted in experts ids in ascending order) in column major layout
    remapped_key_slot[tid] = source_topK_id * num_rows + source_token_id;
    
    cg::sync(cta);

    // Store : store the row to
    row_id_map[out_idx] = remapped_key_slot[tid];

    cg::sync(cta);

    // TODO (yiakwy) : invert remap_id_map
    if (inverted) {
    // Load :
    
    
    // Compute :


    // Store :
    }
}

template <typename T, typename TCompute, int kElementsPerAccess, bool hasProb>
__global__ void moe_recover_topK_kernel(const T *input,
                                        T *unpermuted_output,
                                        const int *row_id_map,
                                        const float *prob,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    // each block corresponds to one source token
    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x * blockDim.y)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        FragmentLoadStore frag_load_store;
        FragmentCompute frag_elem;
        FragmentCompute frag_sum;

        int source_row = row_id_map[source_token];
        const T *source_row_ptr = input + source_row * num_cols;

        cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
            frag_load_store, (source_row_ptr + i), true);
        frag_sum = src_converter(frag_load_store);

        if (hasProb)
        {
            frag_sum = frag_sum * s_prob[0];
        }

        for (int k = 1; k < num_topK; k++)
        {
            source_row = row_id_map[k * num_rows + source_token];
            source_row_ptr = input + source_row * num_cols;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_elem = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_elem = frag_elem * s_prob[k];
            }

            for (int e = 0; e < kElementsPerAccess; e++)
            {
                frag_sum.at(e) = frag_sum.at(e) + frag_elem.at(e);
            }
        }

        T *dest_row_ptr = unpermuted_output + source_token * num_cols;
        frag_load_store = dst_converter(frag_sum);
        *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());
    }
}

template <typename T,
          typename TCompute,
          int  kElementsPerAccess,
          int  topKTile,
          bool hasProb>
__global__ void moe_permute_topK_kernel(const T *input_bwd,
                                        const T *input_fwd,
                                        T *act_grad,
                                        const float *prob,
                                        float *prob_grad,
                                        const int *row_id_map,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    float accum[topKTile] = {0.0f};
    FragmentLoadStore frag_load_store;

    const T *source_row_ptr = input_bwd + source_token * num_cols;
    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
            frag_load_store, (source_row_ptr + i), true);
        FragmentCompute frag_src = src_converter(frag_load_store);

        int index = source_token;

        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            int dest_row = row_id_map[index];
            index += num_rows;

            if (hasProb)
            {
                frag_load_store = dst_converter(frag_src * s_prob[k]);
            }
            else
            {
                frag_load_store = dst_converter(frag_src);
            }

            T *dest_row_ptr = act_grad + dest_row * num_cols;
            *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());

            if (hasProb)
            {
                const T *input_fwd_ptr = input_fwd + dest_row * num_cols;
                cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                    frag_load_store, (input_fwd_ptr + i), true);
                FragmentCompute frag_input_fwd = src_converter(frag_load_store);

                for (int e = 0; e < kElementsPerAccess; e++)
                {
                    accum[k] += float(frag_src.at(e) * frag_input_fwd.at(e));
                }
            }
        }
    }

    if (hasProb)
    {
        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            for (int mask = 16; mask > 0; mask /= 2)
            {
                accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
            }
        }

        if (tid == 0)
        {
            for (int k = 0; k < topKTile; k++)
            {
                if (k == num_topK) break;
                prob_grad[source_token * num_topK + k] = accum[k];
            }
        }
    }
}

template <typename T,
          typename TCompute,
          int kElementsPerAccess,
          int topKTile,
          bool hasProb>
__global__ void moe_permute_topK_kernel_v2(const T *input_bwd,/*original input*/
                                           const T *input_fwd,
                                           T *act_grad, /*output*/
                                           const float *prob,
                                           float *prob_grad,
                                           const int *row_id_map,
                                           const int num_rows,
                                           const int num_topK,
                                           const int num_cols,
                                           bool inverted = false /* row_id_map is dest row id <source_topK_id, source_token_id> of routed tokens to source tokens row id position map if inverted is false*/)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    // TODO (yiakwy) : 4 elements (hdim) length of SRAM array
    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    /*
    // TODO (yiakwy) : make sure TCompute is not equal to T, i.e., should not happen convert(float*, float*)
    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;
     */

    /*
    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;
     */

    // BLOCK_SIZE=1, launch (BLOCK_SIZE/*blockDim.x*/, kThreads/*blockDim.y*/) threads 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid = threadIdx.x + blockDim.x * threadIdx.y;

    if (hasProb)
    {
        /*
        for (int i = tid; i < num_topK; i += blockDim.x)
        {
            // TODO (yiakwy) : cast if T is not float
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
         */
        assert(0);
        __syncthreads();
    }

    float accum[topKTile] = {0.0f}; // upto 128 elements length of on chip memory tile
    FragmentLoadStore frag_load_store;

    if (inverted) {
        assert(0);
        __syncthreads();
    } else {
        const int& dest_row = y + blockIdx.y;
        
        // Load-1 : load source rows and dest rows pair
        const int source_row = row_id_map[dest_row];

        const T *source_row_ptr = input_bwd + source_row * num_cols;
        T *dest_row_ptr = act_grad + dest_row * num_cols;

        using global_load_tile = cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>;
        using global_store_tile = cutlass::arch::global_store<FragmentLoadStore, sizeof(FragmentLoadStore)>;
        
        for (int i=0; i < num_cols; i += kElementsPerAccess * blockDim.x) {
            // Load-2 : load kElementsPerAccess from source rows at source_row_ptr + offset (< num_cols)
            global_load_tile(frag_load_store, source_row_ptr + i * kElementsPerAccess, false);

            // Store : store kElementsPerAccess to 
            global_store_tile(frag_load_store, dest_row_ptr + i * kElementsPerAccess, false);
        }
    }

    if (hasProb)
    {
        assert(0);
        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            for (int mask = 16; mask > 0; mask /= 2)
            {
                accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
            }
        }

        if (tid == 0)
        {
            for (int k = 0; k < topKTile; k++)
            {
                if (k == num_topK) break;
                // prob_grad[source_token * num_topK + k] = accum[k];
            }
        }
    }
}

template <typename T, typename TCompute, bool FWD, int kElementsPerAccess>
void moe_permute_topK_kernel_launcher(
    const T *input,
    T *output,
    const int *sorted_row_id,
    int *row_id_map,
    const float *prob,
    const int num_rows,
    const int num_topK,
    const int num_cols,
    cudaStream_t stream,
    float *prob_grad = nullptr,
    const T *input_fwd = nullptr)
{
    if (FWD)
    {
        if (prob_grad == nullptr)
        {
            // permute_topK fwd
            int threads = 64;
            int blocks = (num_rows * num_topK + threads - 1) / threads;
            moe_permute_topK_row_map<<<blocks, threads, 0, stream>>>(
                sorted_row_id,
                row_id_map,
                num_rows,
                num_topK);

            blocks = num_rows;
            threads = std::min(num_cols / kElementsPerAccess, 1024);
            moe_permute_topK_kernel<T, T, kElementsPerAccess, 128, false><<<blocks, threads, 0, stream>>>(
                input,
                nullptr,
                output,
                nullptr,
                nullptr,
                row_id_map,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK bwd
            int blocks = num_rows;
            int threads = 32;
            size_t smem_bytes = num_topK * sizeof(TCompute);

            if (num_topK == 1)
            {
                moe_permute_topK_kernel<T, T, kElementsPerAccess, 1, false><<<blocks, threads, 0, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 8)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 16)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 16, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 32)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 32, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 64)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 64, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 128)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else
            {
                throw std::runtime_error("num_topK cannot exceed 128.");
            }
        }
    }
    else
    {
        int blocks = num_rows;
        int threads = std::min(num_cols / kElementsPerAccess, 1024);
        size_t smem_bytes = num_topK * sizeof(TCompute);

        if (num_topK == 1)
        {
            // permute_topK bwd with topK==1
            moe_recover_topK_kernel<T, T, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else if (prob == nullptr)
        {
            // permute_topK bwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK fwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, true><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
    }
}

template <typename T, typename TCompute, bool FWD, int kElementsPerAccess>
void moe_permute_topK_kernel_launcher_v2(
    const T *input,
    T *output,
    const int *sorted_row_id,
    int *row_id_map,
    const float *prob,
    const int num_rows, /*num_tokens*/
    const int num_topK,
    const int num_cols, /*hidden_sise*/
    cudaStream_t stream,
    float *prob_grad = nullptr,
    const T *input_fwd = nullptr)
{
    void (*typed_moe_permute_topK_kernel)(const float */*input_0*/, const float */*input_1*/, float */*output*/, const float */*prob*/, float */*prob_grad*/, const int* /*src2dest_row_id_map*/, const int /*rows*/, const int /*topK*/, const int /*cols*/, bool) = nullptr;
    if (FWD)
    {
        if (prob_grad == nullptr)
        {
            // permute_topK fwd
            // TODO (yiakwy) : only 2 warps per block per SM for float32 data, each thread only process 1 element
            int threads = 64;
            int blocks = (num_rows * num_topK + threads - 1) / threads;
            size_t smem_bytes = 2 * num_topK * sizeof(int32_t);
            moe_permute_topK_row_map_v2<<<blocks, threads, smem_bytes, stream>>>(
                sorted_row_id,
                row_id_map,
                num_rows,
                num_topK);

            int BLOCK_SIZE_TOKEN_DIM = 1;
            const int elementsPerThread = std::max(num_cols / (kElementsPerAccess * 32), 1);
            blocks = (num_rows + BLOCK_SIZE_TOKEN_DIM - 1) / BLOCK_SIZE_TOKEN_DIM;
             /* NOTE(yiakwy) : in the simple case of 4 tokens x 128 hdim float dtype input => blocks=4, 1 warp(32 threads)*/
            threads = std::min(num_cols / (kElementsPerAccess * elementsPerThread), 1024);

            dim3 dimBlock(threads, BLOCK_SIZE_TOKEN_DIM, 1);
            dim3 dimGrid(1, blocks, 1);

            typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 128, false>;

            typed_moe_permute_topK_kernel<<<dimGrid, threads, 0, stream>>>(
                input,
                nullptr,
                output,
                nullptr,
                nullptr,
                row_id_map,
                num_rows,
                num_topK,
                num_cols,
                false);
        }
        else /*prob_grad*/
        {
            throw std::underflow_error("Not Implemented yet!");

            // unpermute_topK bwd
            int blocks = num_rows;
            int threads = 32;
            size_t smem_bytes = num_topK * sizeof(TCompute);

            if (num_topK == 1)
            {
                typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 1, false>;
            }
            else if (num_topK <= 8)
            {
                typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 8, false>;
            }
            else if (num_topK <= 16)
            {
                typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 16, false>;
            }
            else if (num_topK <= 32)
            {
                typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 32, false>;
            }
            else if (num_topK <= 64)
            {
                typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 64, false>;
            }
            else if (num_topK <= 128)
            {
                typed_moe_permute_topK_kernel = &moe_permute_topK_kernel_v2<T, T, kElementsPerAccess, 128, false>;
            }
            else
            {
                throw std::runtime_error("num_topK cannot exceed 128.");
            }

            typed_moe_permute_topK_kernel<<<blocks, threads, smem_bytes, stream>>>(
                input,
                input_fwd,
                output,
                prob,
                prob_grad,
                row_id_map,
                num_rows,
                num_topK,
                num_cols,
                false);
        }
    }
    else
    {

        // TODO (yiakwy) : optimize bwd op

        int blocks = num_rows;
        int threads = std::min(num_cols / kElementsPerAccess, 1024);
        size_t smem_bytes = num_topK * sizeof(TCompute);


        if (num_topK == 1)
        {
            // permute_topK bwd with topK==1
            moe_recover_topK_kernel<T, T, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else if (prob == nullptr)
        {
            // permute_topK bwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK fwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, true><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Permute_topK OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num,
    bool                use_fast_permute)
{
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // initialize the workspace on the first run
    if (workspace.empty()) {
        // See https://github.com/pytorch/pytorch/blob/06b845dedca77ed3be756efc1176c4594da2fa80/c10/core/TensorOptions.h#L139C1-L139C29
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options); // TODO (yiakwy) : ?
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        // Warm up (?), https://github.com/dmlc/cub/blob/05eb57faa0a4cac37c2a86fdf4b4dc865a95a1a3/cub/device/device_radix_sort.cuh#L248
        cub::DeviceRadixSort::SortPairs(nullptr/*temp_storage_ptr*/, temp_storage_bytes/*length of temp storage ptr*/,
                                        temp_ptr/*d_keys_in*/, temp_ptr/*d_keys_out*/,
                                        temp_ptr/*d_values_in*/, temp_ptr/*d_values_out*/, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    // NOTE(yiakwy) accelerated by GPU RadixSort, https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK);

    // Activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor permuted_output =
        torch::empty({num_tokens * num_topK, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map =
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // TODO(yiakwy) : better to use a dtype converter and dtype size calculator
    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        if (use_fast_permute) {
            moe_permute_topK_kernel_launcher_v2<dType, dTypeCompute, true, 4>(
                input_ptr,
                permuted_output_ptr,
                sorted_row_id_ptr,
                row_id_map_ptr,
                nullptr,
                num_tokens,
                num_topK,
                num_cols,
                stream
            );
        } else {
        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);
        }

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Unpermute_topK OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor moe_recover_topK_op(
    torch::Tensor  input,
    torch::Tensor  row_id_map,
    torch::Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    // Handle optional tensors; replace `None` with empty tensors
    // Tensor prob = prob_opt.value_or(torch::Tensor());

    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

std::tuple<torch::Tensor, torch::Tensor> moe_recover_topK_bwd_op(
    Tensor  input_bwd,
    Tensor  input_fwd,
    Tensor  row_id_map,
    Tensor  prob)
{
    const int num_tokens = prob.size(0);
    const int num_topK = prob.size(1);
    const int num_cols = input_bwd.size(1);

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = get_ptr<float>(prob);

    // activations type
    const at::ScalarType _st = input_bwd.scalar_type();

    // Output buffer alloc
    Tensor act_grad =
        torch::empty({num_tokens * num_topK, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor prob_grad =
        torch::empty({num_tokens, num_topK}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    float *prob_grad_ptr = get_ptr<float>(prob_grad);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(act_grad, prob_grad);
}


}  // namespace grouped_gemm
