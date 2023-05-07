add_executable(cugemm_v1 cugemm_v1.cu)
set_target_properties(cugemm_v1 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(cugemm_v1 ${CUDA_cublas_LIBRARY})
target_compile_options(cugemm_v1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)

add_executable(cugemm_v2 cugemm_v2.cu)
set_target_properties(cugemm_v2 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(cugemm_v2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)

add_executable(cugemm_v3 cugemm_v3.cu)
set_target_properties(cugemm_v3 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(cugemm_v3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)