#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32

#include <mxnet/base.h>
#include <math.h>

namespace mxnet
{
namespace op
{

// __global__ void matrixMultiplyShared(float *A, float *B, float *C,
//                                     int numAColumns, int numCRows, int numCColumns) 
// {
//   __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  
//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   float sum = 0;
//   for(int p = 0; p < ceil((float)numAColumns / TILE_WIDTH); p++ ){
//     if(row < numCRows && (p * TILE_WIDTH + threadIdx.x) < numAColumns){
//       Mds[threadIdx.y][threadIdx.x] = A[row * numAColumns + p * TILE_WIDTH + threadIdx.x];
//     }else{
//       Mds[threadIdx.y][threadIdx.x] = 0;
//     }
//     if(col < numCColumns && (p * TILE_WIDTH + threadIdx.y < numAColumns)){
//       Nds[threadIdx.y][threadIdx.x] = B[col * numAColumns + p * TILE_WIDTH + threadIdx.y];
//     }else{
//       Nds[threadIdx.y][threadIdx.x] = 0;
//     }
    
//     __syncthreads();
    
//     for(int k = 0; k < TILE_WIDTH; k++){
//       sum += Mds[threadIdx.y][k]* Nds[threadIdx.x][k];
//     }
//     __syncthreads();
//   }
//   if(row < numCRows && col < numCColumns){
//     C[row * numCColumns + col] = sum;
//   }
  
// }
__global__ void matrixMultiplyShared(float *A, float *B, float *C,int numAColumns,
                                     int numCRows, int numCColumns) {

  float PValue = 0;
  int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (TILE_WIDTH + numAColumns - 1)/TILE_WIDTH; k++) {

      if (k*TILE_WIDTH + threadIdx.x < numAColumns && Row < numCRows) 
        As[threadIdx.y][threadIdx.x] = A[Row*numAColumns + k*TILE_WIDTH + threadIdx.x];
      else 
        As[threadIdx.y][threadIdx.x] = 0.0;

      if (Col < numCColumns)  
        Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*numCColumns + Col];
      else 
        Bs[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int n = 0; n < TILE_WIDTH; ++n) 
        PValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

      __syncthreads();

  }

  if (Row < numCRows && Col < numCColumns) 
    C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns)+(blockIdx.x*blockDim.x)+threadIdx.x]=PValue;

}


__global__ void unrollKernel(int C, int H, int W, int K, float* X, float* X_unroll) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int c, s, h_out, w_out, h_unroll, w_base, p, q, w_unroll;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int H_unroll = C*K*K;
    if (idx < C * W_unroll) {
        c = idx / W_unroll;
        s = idx % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        for(p = 0; p < K; p++) {
            for(q = 0; q < K; q++) {
                w_unroll = w_base + p * K + q; 
                // X_unroll[h_unroll * H_unroll + w_unroll ] = X[(c) * (H * W) + (h_out+p) * (W) + w_out+q];
                X_unroll[h_unroll+W_unroll*w_unroll] = X[(c) * (H * W) + (h_out+p) * (W) + w_out+q];
            }
        }
    }
}

void gemm(float* Kernel, float* X_unrolled,  float* Y, int CKK, int M, int HW) {
    // int gridDimY = ceil(1.0 * M / TILE_WIDTH);
    // int gridDimX = ceil(1.0 * HW / TILE_WIDTH);
    dim3 gridDim (ceil(1.0 * HW / TILE_WIDTH),  ceil(1.0 * M / TILE_WIDTH), 1);
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel, X_unrolled, Y, CKK, M, HW);
}

void unroll_gpu(int C, int H, int W, int K, float* X, float* X_unroll)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int num_blocks = ceil((1.0 *C * H_out * W_out) / BLOCK_SIZE);
    unrollKernel<<<num_blocks, BLOCK_SIZE>>>(C, H, W, K, X, X_unroll);
}



/* 
  This function is called by new-inl.h
  Any code you write should be executed by this function.
  For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
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
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    float* Y = y.dptr_;
    float* X = x.dptr_;
    float* Kernel = k.dptr_;
    int W_unroll = C * K * K;
    int H_unroll = H_out * W_out;
    
    float* X_unrolled;
    cudaMalloc(&X_unrolled, sizeof(float)* W_unroll * H_unroll);
    for (int b = B-1; b >= 0; b--) {
        unroll_gpu(C, H, W, K, X + b*C*H*W , X_unrolled);
        gemm(Kernel,  X_unrolled,  Y + b * M * H_out * W_out,  W_unroll,  M,  H_unroll);
    }
    cudaFree(X_unrolled);

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