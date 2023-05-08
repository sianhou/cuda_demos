#include "test.cuh"
#include "fstream"
#include "iomanip"

#define smA(i, j) shared_a[j][i]
#define smB(i, j) shared_b[j][i]

template<int BLK, int STRIDE>
__global__ void sgemm(const float *a, const float *b, float *c, int M, int N, int K) {

    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int lty = threadIdx.y;
    int ltx = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    constexpr int STEP = STRIDE * BLK;

    __shared__ __align__(16 * 1024) float shared_a[STEP][STEP];
    __shared__ __align__(16 * 1024) float shared_b[STEP][STEP];

    const float *ptr_a = a + bx * STEP;
    const float *ptr_b = b + by * STEP * K;
    float *ptr_c = c + bx * STEP + by * STEP * M;
    float sum[STRIDE][STRIDE] = {0.f};

    for (int kk = 0; kk < K; kk += STEP) {

        for (int j = 0; j < STRIDE; ++j) {
            for (int i = 0; i < STRIDE; ++i) {
                smA(i * BLK + tx, j * BLK + ty) = ptr_a[(j * BLK + ty) * M + i * BLK + tx];
                smB(i * BLK + tx, j * BLK + ty) = ptr_b[(j * BLK + ty) * K + i * BLK + tx];
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < STRIDE; ++j) {
            for (int i = 0; i < STRIDE; ++i) {
                for (int b = 0; b < STEP; ++b) {
                    sum[j][i] += smA()

                        shared_a[j * BLK + ty][b] * shared_b[b][i * BLK + tx];
                }
            }
        }
        __syncthreads();

        ptr_a += STEP;
        ptr_b += STEP * N;
    }
    for (int j = 0; j < STRIDE; ++j) {
        for (int i = 0; i < STRIDE; ++i) {
            ptr_c[(j * BLK + ty) * K + i * BLK + tx] = sum[j][i];
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

