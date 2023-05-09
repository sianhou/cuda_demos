#ifndef SJS_SGEMM_TEST_CUH_
#define SJS_SGEMM_TEST_CUH_

#include "random"
#include "iostream"

#include "cuda_runtime.h"
#include "cublas_v2.h"

struct Result {
  int size;
  float elapsed_cublas, elapsed_sgemm;
  float gflops_cublas, gflops_sgemm;
};

template<typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
                static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

using Ftype = void (*)(const float *, const float *, float *, int, int, int);

struct Test {
  Test(Ftype sgemm, int m, int n, int k) {

      M = m;
      N = n;
      K = k;

      ops = 2.0*m*n*k*1.0e-09;

      this->sgemm = sgemm;

      // init host
      h_A = new float[M*K];
      h_B = new float[K*N];
      h_C = new float[M*N];
      h_R = new float[M*N];

      std::uniform_real_distribution<double> u(-1, 1);
      std::default_random_engine e(time(NULL));

      for (auto i = 0; i < M*K; ++i) {
          h_A[i] = u(e);
      }

      for (auto i = 0; i < K*N; ++i) {
          h_B[i] = u(e);
      }

      for (auto i = 0; i < M*N; ++i) {
          h_C[i] = 0.0f;
      }

      for (auto i = 0; i < M*N; ++i) {
          h_R[i] = 0.0f;
      }

      // init device
      if ((err_ = cudaMalloc((void **)&d_A, M*K*sizeof(float)))!=cudaSuccess) {
          std::cout << "Failed to allocate device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }

      if ((err_ = cudaMalloc((void **)&d_B, K*N*sizeof(float)))!=cudaSuccess) {
          std::cout << "Failed to allocate device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }

      if ((err_ = cudaMalloc((void **)&d_C, M*N*sizeof(float)))!=cudaSuccess) {
          std::cout << "Failed to allocate device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }

      if ((err_ = cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice))!=cudaSuccess) {
          std::cout << "Failed to copy dato to device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }

      if ((err_ = cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice))!=cudaSuccess) {
          std::cout << "Failed to copy dato to device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }

      if ((err_ = cudaMemcpy(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice))!=cudaSuccess) {
          std::cout << "Failed to copy dato to device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }
  }

  ~Test() {
      delete[] h_A;
      delete[] h_B;
      delete[] h_C;
      delete[] h_R;

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);

  }

  void RunCublas(int niter) {

      watch.clear();
      gflops.clear();

      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));

      cudaEvent_t start, stop;
      checkCudaErrors(cudaEventCreate(&start));
      checkCudaErrors(cudaEventCreate(&stop));

      const int lda = K, ldb = N, ldc = N;
      float alpha = 1.0, beta = 0.0;

      for (int it = 0; it < niter; ++it) {
          checkCudaErrors(cudaEventRecord(start, NULL));
          cublasSgemm(handle,
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      M,
                      N,
                      K,
                      reinterpret_cast<const float *>(&alpha),
                      reinterpret_cast<const float *>(d_A),
                      lda,
                      reinterpret_cast<const float *>(d_B),
                      ldb,
                      reinterpret_cast<const float *>(&beta),
                      d_C,
                      ldc);
          checkCudaErrors(cudaEventRecord(stop, NULL));
          checkCudaErrors(cudaEventSynchronize(stop));
          float msec = 0.0f;
          checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
          watch.push_back(msec);
          gflops.push_back(ops/(msec/1000.0f));
      }

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      checkCudaErrors(cublasDestroy(handle));
  }

  void RunSgemm(dim3 grid, dim3 block, int niter) {

      watch.clear();
      gflops.clear();

      cudaEvent_t start, stop;
      checkCudaErrors(cudaEventCreate(&start));
      checkCudaErrors(cudaEventCreate(&stop));

      for (int it = 0; it < niter; ++it) {
          checkCudaErrors(cudaEventRecord(start, NULL));
          sgemm<<<grid, block>>>(reinterpret_cast<const float *>(d_A),
                                 reinterpret_cast<const float *>(d_B),
                                 d_C,
                                 M,
                                 N,
                                 K);
          checkCudaErrors(cudaEventRecord(stop, NULL));
          checkCudaErrors(cudaEventSynchronize(stop));
          float msec = 0.0f;
          checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
          watch.push_back(msec);
          gflops.push_back(ops/(msec/1000.0f));
      }

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
  }

  void CheckResult(dim3 grid, dim3 block) {
      // run cublas
      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      const int lda = K, ldb = N, ldc = N;
      float alpha = 1.0, beta = 0.0;
      cublasSgemm(handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  M,
                  N,
                  K,
                  reinterpret_cast<const float *>(&alpha),
                  reinterpret_cast<const float *>(d_A),
                  lda,
                  reinterpret_cast<const float *>(d_B),
                  ldb,
                  reinterpret_cast<const float *>(&beta),
                  d_C,
                  ldc);
      if ((err_ = cudaMemcpy(h_R, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost))!=cudaSuccess) {
          std::cout << "Failed to copy data from device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }
      checkCudaErrors(cublasDestroy(handle));

      // run sgemm
      if ((err_ = cudaMemcpy(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice))!=cudaSuccess) {
          std::cout << "Failed to copy data to device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }
      sgemm<<<grid, block>>>(reinterpret_cast<const float *>(d_A),
                             reinterpret_cast<const float *>(d_B),
                             d_C,
                             M,
                             N,
                             K);
      if ((err_ = cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost))!=cudaSuccess) {
          std::cout << "Failed to copy data from device memory: "
                    << cudaGetErrorString(err_) << std::endl;
          exit(EXIT_FAILURE);
      }

      // check
      for (int i = 0; i < M*N; ++i) {
          if (fabs(h_R[i] - h_C[i]) > 1e-3) {
              std::cout << "Failed to pass result check: " << std::endl
                        << "index = " << i << std::endl
                        << "cublas = " << h_R[i] << std::endl
                        << "sgemm = " << h_C[i] << std::endl;
              exit(EXIT_FAILURE);
          }
      }
  }

  int M, N, K;
  float *d_A, *d_B, *d_C;
  float *h_A, *h_B, *h_C, *h_R;
  cudaError_t err_;
  Ftype sgemm;
  float ops;
  std::vector<float> watch, gflops;
};

#endif //SJS_SGEMM_TEST_CUH_
