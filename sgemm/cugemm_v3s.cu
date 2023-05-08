#include "iostream"
#include "vector"
#include "iomanip"

#include "test.cuh"

template<int BLK, int STRIDE>
__global__ void sgemm(const float *a, const float *b, float *c, int M, int N, int K) {
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;

  constexpr int STEP = STRIDE*BLK;

  __shared__ float shared_a[STEP][STEP];
  __shared__ float shared_b[STEP][STEP];

  const float *ptr_a = a + by*STEP*K;
  const float *ptr_b = b + bx*STEP;
  float *ptr_c = c + by*STEP*N + bx*STEP;
  float sum[STRIDE][STRIDE] = {0.f};

  for (int kk = 0; kk < K; kk += STEP) {

	for (int j = 0; j < STRIDE; ++j) {
	  for (int i = 0; i < STRIDE; ++i) {
		shared_a[j*BLK + ty][i*BLK + tx] = ptr_a[(j*BLK + ty)*K + i*BLK + tx];
		shared_b[j*BLK + ty][i*BLK + tx] = ptr_b[(j*BLK + ty)*K + i*BLK + tx];
	  }
	}
	__syncthreads();
#pragma unroll

	for (int j = 0; j < STRIDE; ++j) {
	  for (int i = 0; i < STRIDE; ++i) {
		for (int b = 0; b < STEP; ++b) {
		  sum[j][i] += shared_a[j*BLK + ty][b]*shared_b[b][i*BLK + tx];
		}
	  }
	}
	__syncthreads();

	ptr_a += STEP;
	ptr_b += STEP*N;
  }
  for (int j = 0; j < STRIDE; ++j) {
	for (int i = 0; i < STRIDE; ++i) {
	  ptr_c[(j*BLK + ty)*K + i*BLK + tx] = sum[j][i];
	}
  }
  __syncthreads();
}

int main() {
  {
	std::vector<float> time;
	MultiSizeTest<4, 4, 2, 2>(sgemm<4, 2>, time);
  }
  {
	std::vector<float> time;
	MultiSizeTest<8, 8, 2, 2>(sgemm<8, 2>, time);
  }
  {
	std::vector<float> time;
	MultiSizeTest<16, 16, 2, 2>(sgemm<16, 2>, time);
  }
  {
	std::vector<float> time;
	MultiSizeTest<32, 32, 2, 2>(sgemm<32, 2>, time);
  }
}

