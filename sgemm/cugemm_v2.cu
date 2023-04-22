#include "iostream"
#include "vector"
#include "iomanip"

#include "test.cuh"

template<int BLK>
__global__ void sgemm(const float *a, const float *b, float *c, int M, int N, int K) {
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;

  __shared__ float shared_a[BLK][BLK];
  __shared__ float shared_b[BLK][BLK];

  const float *ptr_a = a + by*BLK*K;
  const float *ptr_b = b + bx*BLK;
  float sum = 0.f;

  for (int kk = 0; kk < K; kk += BLK) {
	shared_a[ty][tx] = ptr_a[ty*K + tx];
	shared_b[ty][tx] = ptr_b[ty*K + tx];
	__syncthreads();

#pragma unroll
	for (int i = 0; i < BLK; ++i) {
	  sum += shared_a[ty][i]*shared_b[i][tx];
	}
	__syncthreads();

	ptr_a += BLK;
	ptr_b += BLK*N;
  }

  c[(BLK*by + ty)*N + BLK*bx + tx] = sum;
}

int main() {
  {
	std::vector<float> time;
	MultiSizeTest<4, 4, 1, 1>(sgemm<4>, time);
  }
  {
	std::vector<float> time;
	MultiSizeTest<8, 8, 1, 1>(sgemm<8>, time);
  }
  {
	std::vector<float> time;
	MultiSizeTest<16, 16, 1, 1>(sgemm<16>, time);
  }
  {
	std::vector<float> time;
	MultiSizeTest<32, 32, 1, 1>(sgemm<32>, time);
  }
}

