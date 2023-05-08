#include "test.cuh"
#include "fstream"
#include "iomanip"

#define smA(i, j) shared_a[j][i]
#define smB(i, j) shared_b[j][i]

template<int BLK>
__global__ void sgemm(const float *a, const float *b, float *c, int M, int N, int K) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int lty = threadIdx.y;
    int ltx = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    __shared__ float shared_a[BLK][BLK];
    __shared__ float shared_b[BLK][BLK];

    const float *ptr_a = a + bx * BLK;
    const float *ptr_b = b + by * BLK * K;
    float sum = 0.f;

    for (int kk = 0; kk < K; kk += BLK) {
        smA(ltx, lty) = ptr_a[ltx + lty * M];
        smB(ltx, lty) = ptr_b[ltx + lty * K];
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLK; ++i) {
            sum += smA(ltx, i) * smB(i, lty);
        }
        __syncthreads();

        ptr_a += BLK * M;
        ptr_b += BLK;
    }
    c[tx + ty * M] = sum;
}

template<int BLK>
Result test_cugemm(int size, int blk, int niter) {

    int M = size, N = size, K = size;
    dim3 grid, block;
    Result res;
    float sum_of_time, sum_of_gfops;
    res.size = size;

    Test test(sgemm<BLK>, M, N, K);

    block.y = blk;
    block.x = blk;

    grid.y = (M + block.y - 1) / block.y;
    grid.x = (N + block.x - 1) / block.x;

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
    res.elapsed_cublas = sum_of_time / niter;
    res.gflops_cublas = sum_of_gfops / niter;

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
    res.elapsed_sgemm = sum_of_time / niter;
    res.gflops_sgemm = sum_of_gfops / niter;

    return res;
}

int main() {
    Result res;
    std::ofstream ofs("sgemm_v2.txt");

    for (int s = 1024; s <= 1024; s += 32) {
        res = test_cugemm<16>(s, 16, 10);

        ofs << std::setw(4) << res.size << " ";
        ofs << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        ofs << std::setw(8) << res.elapsed_cublas << " ";
        ofs << std::setw(8) << res.gflops_cublas << " ";
        ofs << std::setw(8) << res.elapsed_sgemm << " ";
        ofs << std::setw(8) << res.gflops_sgemm << std::endl;
    }

    ofs.close();
}

