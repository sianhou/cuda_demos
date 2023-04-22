//
// Created by sian on 23-4-22.
//

#ifndef SJS_SGEMM_TEST_CUH_
#define SJS_SGEMM_TEST_CUH_

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
class TestGemm {
 public:
  TestGemm(int M, int N, int K, FUNC sgemm) : M(M), N(N), K(K) {
	block.y = BLKY;
	block.x = BLKX;
	auto tile_y = BLKY*STRIDEY;
	auto tile_x = BLKX*STRIDEX;
	grid.y = (N + tile_y - 1)/tile_y;
	grid.x = (M + tile_x - 1)/tile_x;
	this->sgemm = sgemm;
  }

  void WarmUp();
  float Run(int n_iter);
  void InitHost();
  void InitDevice();
  void Free();

  int M, N, K;
  dim3 grid, block;
  float *d_A, *d_B, *d_C;
  float *h_A, *h_B, *h_C;
  cudaError_t err_;
  FUNC sgemm;
};

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
void TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC>::InitHost() {
  h_A = new float[M*K];
  h_B = new float[K*N];
  h_C = new float[M*N];

  for (auto i = 0; i < M*K; ++i) {
	h_A[i] = 1.0f;
  }

  for (auto i = 0; i < K*N; ++i) {
	h_B[i] = 1.0f;
  }

  for (auto i = 0; i < M*N; ++i) {
	h_C[i] = 0.0f;
  }
}

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
void TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC>::InitDevice() {
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

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
void TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC>::Free() {
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
void TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC>::WarmUp() {

  this->sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int j = 0; j < M; ++j) {
	for (int i = 0; i < N; ++i) {
	  if (fabs(h_C[j*N + i] - 1.0f*K) > 1e-6) {
		std::cout << "error in sgemm: " << BLKX << " x " << BLKY << std::endl;
		std::cout << "j x i = " << j << " x " << i << std::endl;
		std::cout << "C =  " << h_C[j*N + i] << std::endl;
		exit(0);
	  }
	}
  }
}

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
float TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC>::Run(int n_iter) {
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

//    std::cout << " ----------- run test() ----------- " << std::endl;

  // 记录开始时刻的时间戳
  cudaEventRecord(start, 0);
  // Do Something

  for (int i = 0; i < n_iter; ++i) {
	this->sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  }

  // 记录结束时刻的时间戳
  cudaEventRecord(stop, 0);
  // 等待事件同步值
  cudaEventSynchronize(stop);

  // 根据开始和结束时刻的时间戳，计算其中所经过的时间
  cudaEventElapsedTime(&elapsedTime, start, stop);
  // 打印时间
//    printf("Average run time: %6.2f ms\n", elapsedTime / float(n_iter));

  return elapsedTime/float(n_iter);
}

template<int BLKY, int BLKX, int STRIDEY, int STRIDEX, typename FUNC>
void MultiSizeTest(FUNC sgemm, std::vector<float> &time) {
  time.clear();

  {
	TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC> test(256, 256, 256, sgemm);
	test.InitHost();
	test.InitDevice();
	test.WarmUp();
	time.push_back(test.Run(10));
	test.Free();
  }

  {
	TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC> test(512, 512, 512, sgemm);
	test.InitHost();
	test.InitDevice();
	test.WarmUp();
	time.push_back(test.Run(10));
	test.Free();
  }

  {
	TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC> test(1024, 1024, 1024, sgemm);
	test.InitHost();
	test.InitDevice();
	test.WarmUp();
	time.push_back(test.Run(10));
	test.Free();
  }

  {
	TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC> test(2048, 2048, 2048, sgemm);
	test.InitHost();
	test.InitDevice();
	test.WarmUp();
	time.push_back(test.Run(10));
	test.Free();
  }

  {
	TestGemm<BLKY, BLKX, STRIDEY, STRIDEX, FUNC> test(4096, 4096, 4096, sgemm);
	test.InitHost();
	test.InitDevice();
	test.WarmUp();
	time.push_back(test.Run(10));
	test.Free();
  }
  std::cout << std::endl << BLKY << "x" << BLKX << " " << "test results(ms)" << std::endl;
  std::cout << "256     512     1024     2048     4096" << std::endl;
  std::cout << "---------------------------------------" << std::endl;
  for (int i = 0; i < time.size(); ++i) {
	std::cout << std::setw(10) << std::setprecision(2) << time[i];
  }
  std::cout << std::endl;
}

#endif //SJS_SGEMM_TEST_CUH_
