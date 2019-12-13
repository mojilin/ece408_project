#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16
namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h, w, c, p, q;
    const int W_grid = ceil(W_out/(float)TILE_WIDTH);

    n = blockIdx.x;
    m = blockIdx.y;
    int block_row_start = blockIdx.z / W_grid * TILE_WIDTH; 
    int block_col_start = blockIdx.z % W_grid * TILE_WIDTH;
    float acc = 0.;
    int input_tile_width = TILE_WIDTH + K - 1;
    int row_o = block_row_start + threadIdx.x; 
    int col_o = block_col_start + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float shmem[];
    float* X_shared1 = &shmem[0];
    float* W_shared1 = &shmem[input_tile_width * input_tile_width];

    float* X_shared2 = &shmem[input_tile_width * input_tile_width + K * K];
    float* W_shared2 = &shmem[input_tile_width * input_tile_width * 2 + K * K];
    float *X_temp;
    float *W_temp;

    for(c = 0; c < C; c++){
        for(int i = tx; i < K; i += TILE_WIDTH){
            for(int j = ty; j < K; j += TILE_WIDTH){
                W_shared1[i * K + j] = k4d(m, c, i, j);
            }
        }

        for(int i = row_o; i < input_tile_width + block_row_start; i += TILE_WIDTH){
            for(int j = col_o; j < input_tile_width + block_col_start; j += TILE_WIDTH){
                if(i < H && j < W){
                    X_shared1[(i - block_row_start) * input_tile_width + j - block_col_start] = x4d(n, c, i, j);
                }
            }
        }
        __syncthreads();

        acc += X_shared1[(threadIdx.x + 0) * input_tile_width + threadIdx.y + 0] * W_shared1[0 * K + 0]
        + X_shared1[(threadIdx.x + 0) * input_tile_width + threadIdx.y + 1] * W_shared1[0 * K + 1]
        + X_shared1[(threadIdx.x + 0) * input_tile_width + threadIdx.y + 2] * W_shared1[0 * K + 2]
        + X_shared1[(threadIdx.x + 0) * input_tile_width + threadIdx.y + 3] * W_shared1[0 * K + 3]
        + X_shared1[(threadIdx.x + 0) * input_tile_width + threadIdx.y + 4] * W_shared1[0 * K + 4]
        + X_shared1[(threadIdx.x + 1) * input_tile_width + threadIdx.y + 0] * W_shared1[0 * K + 0]
        + X_shared1[(threadIdx.x + 1) * input_tile_width + threadIdx.y + 1] * W_shared1[0 * K + 1]
        + X_shared1[(threadIdx.x + 1) * input_tile_width + threadIdx.y + 2] * W_shared1[0 * K + 2]
        + X_shared1[(threadIdx.x + 1) * input_tile_width + threadIdx.y + 3] * W_shared1[0 * K + 3]
        + X_shared1[(threadIdx.x + 1) * input_tile_width + threadIdx.y + 4] * W_shared1[0 * K + 4]
        + X_shared1[(threadIdx.x + 2) * input_tile_width + threadIdx.y + 0] * W_shared1[0 * K + 0]
        + X_shared1[(threadIdx.x + 2) * input_tile_width + threadIdx.y + 1] * W_shared1[0 * K + 1]
        + X_shared1[(threadIdx.x + 2) * input_tile_width + threadIdx.y + 2] * W_shared1[0 * K + 2]
        + X_shared1[(threadIdx.x + 2) * input_tile_width + threadIdx.y + 3] * W_shared1[0 * K + 3]
        + X_shared1[(threadIdx.x + 2) * input_tile_width + threadIdx.y + 4] * W_shared1[0 * K + 4]
        + X_shared1[(threadIdx.x + 3) * input_tile_width + threadIdx.y + 0] * W_shared1[0 * K + 0]
        + X_shared1[(threadIdx.x + 3) * input_tile_width + threadIdx.y + 1] * W_shared1[0 * K + 1]
        + X_shared1[(threadIdx.x + 3) * input_tile_width + threadIdx.y + 2] * W_shared1[0 * K + 2]
        + X_shared1[(threadIdx.x + 3) * input_tile_width + threadIdx.y + 3] * W_shared1[0 * K + 3]
        + X_shared1[(threadIdx.x + 3) * input_tile_width + threadIdx.y + 4] * W_shared1[0 * K + 4]
        + X_shared1[(threadIdx.x + 4) * input_tile_width + threadIdx.y + 0] * W_shared1[0 * K + 0]
        + X_shared1[(threadIdx.x + 4) * input_tile_width + threadIdx.y + 1] * W_shared1[0 * K + 1]
        + X_shared1[(threadIdx.x + 4) * input_tile_width + threadIdx.y + 2] * W_shared1[0 * K + 2]
        + X_shared1[(threadIdx.x + 4) * input_tile_width + threadIdx.y + 3] * W_shared1[0 * K + 3]
        + X_shared1[(threadIdx.x + 4) * input_tile_width + threadIdx.y + 4] * W_shared1[0 * K + 4];
        
        X_temp = X_shared1;
        X_shared1 = X_shared2;
        X_shared2 = X_temp;

        W_temp = W_shared1;
        W_shared1 = W_shared2;
        W_shared2 = W_temp;
    }

    if (row_o < H_out && col_o < W_out) {
        y4d(n, m, row_o, col_o) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Set the kernel dimensions

    const int W_grid = ceil(W_out/(float)TILE_WIDTH); // number of horizontal tiles per output map
    const int H_grid = ceil(H_out/(float)TILE_WIDTH); // number of vertical tiles per output map
    const int Z = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    // Call the kernel
    //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);

    size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K) * 2;

    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_, w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif