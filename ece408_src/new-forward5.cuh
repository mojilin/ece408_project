#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 16

#include <mxnet/base.h>
#include <math.h>

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *W, float *X, float *Y, int C, int K, int M, int W_out, int H_out) 
{
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int b = blockIdx.z;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int numACol = C*K*K;
  int H_in = H_out + K - 1;
  int W_in = W_out + K - 1;
  
  float result = 0;

  int iter = ceil(numACol/(1.0*TILE_WIDTH));
  
  for (int i = 0; i < iter; i++) {
    int temp_col = i * TILE_WIDTH + tx, temp_row = i * TILE_WIDTH + ty;
    tileA[ty][tx] = 0;
    tileB[ty][tx] = 0;
    int W_m = row;
    int W_c = temp_col/(K*K);
    int W_h = (temp_col%(K*K))/K, W_w = (temp_col%(K*K))%K;
    if (temp_col < numACol && row < M) {
      tileA[ty][tx] = W[W_m*C*K*K + W_c*K*K + W_h*K +W_w];
    } else {
      tileB[ty][tx] = 0;
    }
    int X_b = b;
    int X_c = temp_row/(K*K);
    int X_p = temp_row%(K*K)/K, X_q = (temp_row%(K*K))%K;
    int X_h = col / W_out, X_w = col % W_out;
    if (temp_row < numACol && col < H_out*W_out) {
      tileB[ty][tx] = X[X_b*C*H_in*W_in + X_c*H_in*W_in + (X_h+X_p)*W_in  +X_w + X_q];
    } else {
      tileB[ty][tx] = 0;
    }
    __syncthreads();
    for (int q = 0; q < TILE_WIDTH; q++) {
      result += tileA[ty][q] * tileB[q][tx];
    }
    __syncthreads();
  }
    
  int Y_b = b;
  int Y_m = row;
  int Y_h = col / W_out, Y_w = col % W_out;

  if ((row < M) && (col < W_out*H_out)) {
    Y[Y_b*M*W_out*H_out + Y_m*W_out*H_out + Y_h*W_out + Y_w] = result;
  }
  
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
    int W_unroll = H_out * W_out;
    
    dim3 gridDim (ceil(1.0 * W_unroll / TILE_WIDTH),  ceil(1.0 *  M/ TILE_WIDTH), B);
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
    forward_kernel<<<gridDim, blockDim>>>(Kernel, X, Y, C, K, M, W_out, H_out);
    

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
    // CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif