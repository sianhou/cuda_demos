#include "test.cuh"
#include "fstream"
#include "iomanip"

#define smA(i, j) shared_a[j][i]
#define smB(i, j) shared_b[j][i]

template<int BLK, int STRIDE>
__global__ void sgemm(const float *a, const float *b, float *c, int M, int N, int K) {

    constexpr int STEP = STRIDE*BLK;
    const float *ptr_a = a + blockIdx.x*STEP;
    const float *ptr_b = b + blockIdx.y*STEP*K;
    float *ptr_c = c + blockIdx.x*STEP + blockIdx.y*STEP*M;
    float sum[STRIDE][STRIDE] = {0.f};

    __shared__ __align__(16*1024) float shared_a[STEP][STEP];
    __shared__ __align__(16*1024) float shared_b[STEP][STEP];

    for (int kk = 0; kk < K; kk += STEP) {

        for (int ir = 0; ir < STRIDE; ++ir) {
            for (int ic = 0; ic < STRIDE; ++ic) {
                int row = threadIdx.x + ir*BLK;
                int col = threadIdx.y + ic*BLK;
                smA(row, col) = ptr_a[row + col*M];
                smB(row, col) = ptr_b[row + col*K];
            }
        }
        __syncthreads();

#pragma unroll
        for (int ir = 0; ir < STRIDE; ++ir) {
            for (int ic = 0; ic < STRIDE; ++ic) {
                int row = threadIdx.x + ir*BLK;
                int col = threadIdx.y + ic*BLK;
                for (int b = 0; b < STEP; ++b) {
                    sum[ir][ic] += smA(row, b)*smB(b, col);
                }
            }
        }
        __syncthreads();

        ptr_a += STEP*M;
        ptr_b += STEP;
    }

    for (int ir = 0; ir < STRIDE; ++ir) {
        for (int ic = 0; ic < STRIDE; ++ic) {
            int row = threadIdx.x + ir*BLK;
            int col = threadIdx.y + ic*BLK;
            ptr_c[row + col*M] = sum[ir][ic];
        }
    }
    __syncthreads();
}

template<int BLK, int STRIDE>
Result test_cugemm(int size, int blk, int niter) {

    int M = size, N = size, K = size;
    dim3 grid, block;
    Result res;
    float sum_of_time, sum_of_gfops;
    res.size = size;

    Test test(sgemm<BLK, STRIDE>, M, N, K);

    block.y = blk;
    block.x = blk;

    grid.y = (M + block.y*STRIDE - 1)/(block.y*STRIDE);
    grid.x = (N + block.x*STRIDE - 1)/(block.x*STRIDE);

    std::cout << "M = N = K = " << size << std::endl;
    std::cout << "grid.z x grid.y x grid.x = " << grid.z << " x " << grid.y << " x " << grid.x << std::endl;
    std::cout << "block.z x block.y x block.x = " << block.z << " x " << block.y << " x " << block.x << std::endl;

    // warm up and check out
    test.CheckResult(grid, block);

    // cublas
    test.RunCublas(niter);
    std::cout << std::endl << "cublas:" << std::endl;
    sum_of_time = 0;
    sum_of_gfops = 0;
    for (int i = 0; i < niter; ++i) {
        std::cout << i << ": runtime = " << test.watch[i] << ", gflops = " << test.gflops[i] << std::endl;
        sum_of_time += test.watch[i];
        sum_of_gfops += test.gflops[i];
    }
    res.elapsed_cublas = sum_of_time/niter;
    res.gflops_cublas = sum_of_gfops/niter;

    // sgemm
    test.RunSgemm(grid, block, niter);
    std::cout << std::endl << "sgemm:" << std::endl;
    sum_of_time = 0;
    sum_of_gfops = 0;
    for (int i = 0; i < niter; ++i) {
        std::cout << i << ": runtime = " << test.watch[i] << ", gflops = " << test.gflops[i] << std::endl;
        sum_of_time += test.watch[i];
        sum_of_gfops += test.gflops[i];
    }
    res.elapsed_sgemm = sum_of_time/niter;
    res.gflops_sgemm = sum_of_gfops/niter;

    return res;
}

int main() {
    Result res;
    std::ofstream ofs("sgemm_v4_blk8x8_stride2x2.txt");

    for (int s = 64; s <= 4096; s *= 2) {
        res = test_cugemm<8, 2>(s, 8, 10);

        ofs << std::setw(4) << res.size << " ";
        ofs << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        ofs << std::setw(8) << res.elapsed_cublas << " ";
        ofs << std::setw(8) << res.gflops_cublas << " ";
        ofs << std::setw(8) << res.elapsed_sgemm << " ";
        ofs << std::setw(8) << res.gflops_sgemm << std::endl;
    }

    ofs.close();
}

