#include "iostream"
#include "vector"
#include "iomanip"

#include "test.cuh"

__global__ void sgemm(const float *a, const float *b, float *c, int M, int N, int K) {
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;

    if (tx < M && ty < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += a[tx*K + i]*b[i*K + ty];
        }
        c[tx*K + ty] = sum;
    }
}

int main() {

    {
        Test test(sgemm, 128, 128, 128);
        test.RunCublas(1);
    }
//  {
//	std::vector<float> time;
//	MultiSizeTest<4, 4, 1, 1>(sgemm, time);
//  }
//  {
//	std::vector<float> time;
//	MultiSizeTest<8, 8, 1, 1>(sgemm, time);
//  }
//  {
//	std::vector<float> time;
//	MultiSizeTest<16, 16, 1, 1>(sgemm, time);
//  }
//  {
//	std::vector<float> time;
//	MultiSizeTest<32, 32, 1, 1>(sgemm, time);
//  }
}

