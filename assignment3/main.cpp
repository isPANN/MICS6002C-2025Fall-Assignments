#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>

void prefix_sum_cpu(int *in, int *out, int n) {
    out[0] = in[0];
    for (int i = 1; i < n; i++){
        out[i] = out[i - 1] + in[i];
    }
}

// Task A: Single-block prefix-sum for 10^6 elements
__global__ void task_a_single_block_scan(const int* in, int* out, int n) {
    extern __shared__ int s[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Running prefix sum across chunks
    __shared__ int running_prefix;
    if (tid == 0) running_prefix = 0;
    __syncthreads();
    
    // Process array in chunks of 2*block_size
    int chunk_capacity = block_size * 2;
    for (int chunk_start = 0; chunk_start < n; chunk_start += chunk_capacity) {
        int chunk_end = chunk_start + chunk_capacity;
        if (chunk_end > n) chunk_end = n;
        int chunk_size = chunk_end - chunk_start;
        
        // Load chunk into shared memory with padding
        int ai = tid;
        int bi = tid + block_size;
        int bank_offset_a = ai >> 5;
        int bank_offset_b = bi >> 5;
        
        if (ai < chunk_size) {
            s[ai + bank_offset_a] = in[chunk_start + ai];
        }
        if (bi < chunk_size) {
            s[bi + bank_offset_b] = in[chunk_start + bi];
        }
        __syncthreads();
        
        // up-sweep
        for (int offset = 1; offset < chunk_size; offset <<= 1) {
            int idx = tid * (offset << 1) + (offset << 1) - 1;
            if (idx < chunk_size) {
                int idx_padded = idx + (idx >> 5);
                int idx_offset_padded = (idx - offset) + ((idx - offset) >> 5);
                s[idx_padded] += s[idx_offset_padded];
            }
            __syncthreads();
        }
        
        // Save chunk sum
        __shared__ int chunk_sum;
        if (tid == 0) {
            int last_idx = chunk_size - 1 + ((chunk_size - 1) >> 5);
            chunk_sum = s[last_idx];
            s[last_idx] = 0;
        }
        __syncthreads();
        
        // down-sweep
        for (int offset = chunk_size >> 1; offset >= 1; offset >>= 1) {
            int idx = tid * (offset << 1) + (offset << 1) - 1;
            if (idx < chunk_size) {
                int idx_padded = idx + (idx >> 5);
                int idx_offset_padded = (idx - offset) + ((idx - offset) >> 5);
                int temp = s[idx_offset_padded];
                s[idx_offset_padded] = s[idx_padded];
                s[idx_padded] += temp;
            }
            __syncthreads();
        }
        
        // Convert exclusive to inclusive and add running prefix
        if (ai < chunk_size) {
            out[chunk_start + ai] = s[ai + bank_offset_a] + in[chunk_start + ai] + running_prefix;
        }
        if (bi < chunk_size) {
            out[chunk_start + bi] = s[bi + bank_offset_b] + in[chunk_start + bi] + running_prefix;
        }
        __syncthreads();
        
        // Update running prefix
        if (tid == 0) {
            running_prefix += chunk_sum;
        }
        __syncthreads();
    }
}

// Task B: Two-phase prefix-sum for 10^9 elements
// Phase 1: Each block computes prefix sum of its chunk and saves the total
__global__ void task_b_phase1_block_scan(const int* in, int* out, int* block_sums, int n, int elements_per_block) {
    extern __shared__ int s[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    
    // Calculate this block's data range
    int block_start = bid * elements_per_block;
    int block_end = block_start + elements_per_block;
    if (block_end > n) block_end = n;
    int block_data_size = block_end - block_start;
    
    // Load data into shared memory with padding
    int ai = tid;
    int bi = tid + block_size;
    int bank_offset_a = ai >> 5;
    int bank_offset_b = bi >> 5;
    
    if (ai < block_data_size) {
        s[ai + bank_offset_a] = in[block_start + ai];
    }
    if (bi < block_data_size) {
        s[bi + bank_offset_b] = in[block_start + bi];
    }
    __syncthreads();
    
    // up-sweep
    for (int offset = 1; offset < block_data_size; offset <<= 1) {
        int idx = tid * (offset << 1) + (offset << 1) - 1;
        if (idx < block_data_size) {
            int idx_padded = idx + (idx >> 5);
            int idx_offset_padded = (idx - offset) + ((idx - offset) >> 5);
            s[idx_padded] += s[idx_offset_padded];
        }
        __syncthreads();
    }
    
    // Store block sum
    if (tid == 0) {
        int last_idx = block_data_size - 1 + ((block_data_size - 1) >> 5);
        block_sums[bid] = s[last_idx];
        s[last_idx] = 0;
    }
    __syncthreads();
    
    // down-sweep
    for (int offset = block_data_size >> 1; offset >= 1; offset >>= 1) {
        int idx = tid * (offset << 1) + (offset << 1) - 1;
        if (idx < block_data_size) {
            int idx_padded = idx + (idx >> 5);
            int idx_offset_padded = (idx - offset) + ((idx - offset) >> 5);
            int temp = s[idx_offset_padded];
            s[idx_offset_padded] = s[idx_padded];
            s[idx_padded] += temp;
        }
        __syncthreads();
    }
    
    // Convert exclusive to inclusive and store
    if (ai < block_data_size) {
        out[block_start + ai] = s[ai + bank_offset_a] + in[block_start + ai];
    }
    if (bi < block_data_size) {
        out[block_start + bi] = s[bi + bank_offset_b] + in[block_start + bi];
    }
}

// Phase 2: Scan the block sums to get prefix sums of block sums
__global__ void task_b_phase2_scan_block_sums(int* block_sums, int num_blocks) {
    extern __shared__ int s[];
    int tid = threadIdx.x;
    
    // Load block sums into shared memory with padding
    int bank_offset = tid >> 5;
    if (tid < num_blocks) {
        s[tid + bank_offset] = block_sums[tid];
    }
    __syncthreads();
    
    // Blelloch up-sweep
    for (int offset = 1; offset < num_blocks; offset <<= 1) {
        int idx = tid * (offset << 1) + (offset << 1) - 1;
        if (idx < num_blocks) {
            int idx_padded = idx + (idx >> 5);
            int idx_offset_padded = (idx - offset) + ((idx - offset) >> 5);
            s[idx_padded] += s[idx_offset_padded];
        }
        __syncthreads();
    }
    
    // Clear last element for exclusive scan
    if (tid == 0) {
        int last_idx = num_blocks - 1 + ((num_blocks - 1) >> 5);
        s[last_idx] = 0;
    }
    __syncthreads();
    
    // Blelloch down-sweep
    for (int offset = num_blocks >> 1; offset >= 1; offset >>= 1) {
        int idx = tid * (offset << 1) + (offset << 1) - 1;
        if (idx < num_blocks) {
            int idx_padded = idx + (idx >> 5);
            int idx_offset_padded = (idx - offset) + ((idx - offset) >> 5);
            int temp = s[idx_offset_padded];
            s[idx_offset_padded] = s[idx_padded];
            s[idx_padded] += temp;
        }
        __syncthreads();
    }
    
    // Store exclusive prefix sums
    if (tid < num_blocks) {
        block_sums[tid] = s[tid + bank_offset];
    }
}

// Phase 3: Add block prefix sums to each block's elements
__global__ void task_b_phase3_add_block_prefix(int* out, const int* block_sums, int n, int elements_per_block) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Skip first block (no prefix to add)
    if (bid == 0) return;
    
    // Get this block's prefix sum
    int prefix_sum = block_sums[bid];
    
    // Add prefix to all elements in this block
    int block_start = bid * elements_per_block;
    int block_end = block_start + elements_per_block;
    if (block_end > n) block_end = n;
    
    for (int i = block_start + tid; i < block_end; i += blockDim.x) {
        out[i] += prefix_sum;
    }
}

int main() {
    cudaSetDevice(0);
    
    const int N_A = 1000000;   // 10^6 elements for Task A
    const int N_B = 1000000000; // 10^9 elements for Task B
    
    // Task A: 10^6 elements
    printf("=== Task A: 10^6 elements ===\n");
    int* h_in_A = (int*)malloc(N_A * sizeof(int));
    int* h_out_A = (int*)malloc(N_A * sizeof(int));
    
    // Initialize input array A
    for (int i = 0; i < N_A; i++) {
        h_in_A[i] = i % 10;  // Simple pattern: 0,1,2,3,4,5,6,7,8,9,0,1,2,...
    }

    // CPU version for Task A
    printf("=== CPU Version (Task A) ===\n");
    int* h_out_cpu_A = (int*)malloc(N_A * sizeof(int));
    
    // CPU timing using clock()
    clock_t start_cpu_A = clock();
    prefix_sum_cpu(h_in_A, h_out_cpu_A, N_A);
    clock_t end_cpu_A = clock();
    
    double time_cpu_A = ((double)(end_cpu_A - start_cpu_A)) / CLOCKS_PER_SEC * 1000.0;  // Convert to ms
    printf("CPU Task A - Total time: %.3f ms\n", time_cpu_A);
    
    // Task B: 10^9 elements
    printf("=== Task B: 10^9 elements ===\n");
    int* h_in_B = (int*)malloc(N_B * sizeof(int));
    int* h_out_B = (int*)malloc(N_B * sizeof(int));
    
    // Initialize input array B
    for (int i = 0; i < N_B; i++) {
        h_in_B[i] = i % 10;  // Simple pattern: 0,1,2,3,4,5,6,7,8,9,0,1,2,...
    }

    // CPU version for Task B
    printf("=== CPU Version (Task B) ===\n");
    int* h_out_cpu_B = (int*)malloc(N_B * sizeof(int));
    
    // CPU timing using clock()
    clock_t start_cpu_B = clock();
    prefix_sum_cpu(h_in_B, h_out_cpu_B, N_B);
    clock_t end_cpu_B = clock();
    
    double time_cpu_B = ((double)(end_cpu_B - start_cpu_B)) / CLOCKS_PER_SEC * 1000.0;  // Convert to ms
    printf("CPU Task B - Total time: %.3f ms\n", time_cpu_B);

    // Task A: Single block version with timing
    printf("=== Task A: Single Block Version (10^6 elements) ===\n");
    
    int *d_in_A, *d_out_A;
    cudaMalloc(&d_in_A, N_A * sizeof(int));
    cudaMalloc(&d_out_A, N_A * sizeof(int));
    
    // Create events for Task A timing
    cudaEvent_t start_taskA_total, stop_taskA_total;
    cudaEvent_t start_taskA_kernel, stop_taskA_kernel;
    cudaEventCreate(&start_taskA_total);
    cudaEventCreate(&stop_taskA_total);
    cudaEventCreate(&start_taskA_kernel);
    cudaEventCreate(&stop_taskA_kernel);
    
    // Add padding for bank conflict avoidance
    int shared_mem_size_A = (2048 + (2048 >> 5)) * sizeof(int);
    
    // Start total timing (including ALL memory copies)
    cudaEventRecord(start_taskA_total);
    
    // Host to Device memory copy
    cudaMemcpy(d_in_A, h_in_A, N_A * sizeof(int), cudaMemcpyHostToDevice);
    
    // Start kernel-only timing
    cudaEventRecord(start_taskA_kernel);
    task_a_single_block_scan<<<1, 1024, shared_mem_size_A>>>(d_in_A, d_out_A, N_A);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_taskA_kernel);
    
    // Device to Host memory copy
    cudaMemcpy(h_out_A, d_out_A, N_A * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_taskA_total);
    cudaEventSynchronize(stop_taskA_total);
    
    // Calculate Task A timing
    float time_taskA_total, time_taskA_kernel;
    cudaEventElapsedTime(&time_taskA_total, start_taskA_total, stop_taskA_total);
    cudaEventElapsedTime(&time_taskA_kernel, start_taskA_kernel, stop_taskA_kernel);
    
    printf("Task A - Total time (with memory copy): %.3f ms\n", time_taskA_total);
    printf("Task A - Kernel-only time (without memory copy): %.3f ms\n", time_taskA_kernel);
    
    // Cleanup Task A events
    cudaEventDestroy(start_taskA_total);
    cudaEventDestroy(stop_taskA_total);
    cudaEventDestroy(start_taskA_kernel);
    cudaEventDestroy(stop_taskA_kernel);

    // Task B: Multi-block version with timing
    printf("=== Task B: Multi-Block Version (10^9 elements) ===\n");
    
    int *d_in_B, *d_out_B;
    cudaMalloc(&d_in_B, N_B * sizeof(int));
    cudaMalloc(&d_out_B, N_B * sizeof(int));
    
    // Create events for Task B timing
    cudaEvent_t start_taskB_total, stop_taskB_total;
    cudaEvent_t start_taskB_kernel, stop_taskB_kernel;
    cudaEventCreate(&start_taskB_total);
    cudaEventCreate(&stop_taskB_total);
    cudaEventCreate(&start_taskB_kernel);
    cudaEventCreate(&stop_taskB_kernel);
    
    int* h_out_taskB = (int*)malloc(N_B * sizeof(int));
    
    const int elements_per_block = 2048;  // Each block handles 2048 elements
    const int num_blocks_B = (N_B + elements_per_block - 1) / elements_per_block;
    
    int* d_block_sums_B;
    cudaMalloc(&d_block_sums_B, num_blocks_B * sizeof(int));
    
    // Add padding for bank conflict avoidance
    int shared_mem_size_B = (2048 + (2048 >> 5)) * sizeof(int);
    int shared_mem_phase2_B = (num_blocks_B + (num_blocks_B >> 5)) * sizeof(int);
    
    // Start total timing (including ALL memory copies)
    cudaEventRecord(start_taskB_total);
    
    // Host to Device memory copy
    cudaMemcpy(d_in_B, h_in_B, N_B * sizeof(int), cudaMemcpyHostToDevice);
    
    // Start kernel-only timing
    cudaEventRecord(start_taskB_kernel);
    
    // Phase 1: Each block computes prefix sum of its chunk
    task_b_phase1_block_scan<<<num_blocks_B, 1024, shared_mem_size_B>>>(d_in_B, d_out_B, d_block_sums_B, N_B, elements_per_block);
    
    // Phase 2: Scan the block sums
    task_b_phase2_scan_block_sums<<<1, 1024, shared_mem_phase2_B>>>(d_block_sums_B, num_blocks_B);
    
    // Phase 3: Add block prefix sums to each block
    task_b_phase3_add_block_prefix<<<num_blocks_B, 1024>>>(d_out_B, d_block_sums_B, N_B, elements_per_block);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop_taskB_kernel);
    
    // Device to Host memory copy
    cudaMemcpy(h_out_taskB, d_out_B, N_B * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_taskB_total);
    cudaEventSynchronize(stop_taskB_total);
    
    // Calculate Task B timing
    float time_taskB_total, time_taskB_kernel;
    cudaEventElapsedTime(&time_taskB_total, start_taskB_total, stop_taskB_total);
    cudaEventElapsedTime(&time_taskB_kernel, start_taskB_kernel, stop_taskB_kernel);
    
    printf("Task B - Total time (with memory copy): %.3f ms\n", time_taskB_total);
    printf("Task B - Kernel-only time (without memory copy): %.3f ms\n", time_taskB_kernel);
    
    // Cleanup Task B events
    cudaEventDestroy(start_taskB_total);
    cudaEventDestroy(stop_taskB_total);
    cudaEventDestroy(start_taskB_kernel);
    cudaEventDestroy(stop_taskB_kernel);

    // Print first 20 elements - CPU vs Task A vs Task B
    printf("\n=== Results Comparison ===\n");
    printf("CPU Task A: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_cpu_A[i]);
    }
    printf("\nTask A: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_A[i]);
    }
    printf("\nTask B: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_taskB[i]);
    }
    printf("\n\n");

    // Task C: Concurrent execution using CUDA streams
    printf("=== Task C: Concurrent Execution with CUDA Streams ===\n");
    
    // Create 4 CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    
    // Create CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_taskA1, stop_taskA1, start_taskA2, stop_taskA2;
    cudaEvent_t start_taskB1, stop_taskB1, start_taskB2, stop_taskB2;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_taskA1);
    cudaEventCreate(&stop_taskA1);
    cudaEventCreate(&start_taskA2);
    cudaEventCreate(&stop_taskA2);
    cudaEventCreate(&start_taskB1);
    cudaEventCreate(&stop_taskB1);
    cudaEventCreate(&start_taskB2);
    cudaEventCreate(&stop_taskB2);
    
    // Allocate memory for all 4 tasks
    // TaskA1, TaskA2: 10^6 elements each
    int* d_out_taskA1, *d_out_taskA2;
    int* h_out_taskA1 = (int*)malloc(N_A * sizeof(int));
    int* h_out_taskA2 = (int*)malloc(N_A * sizeof(int));
    
    cudaMalloc(&d_out_taskA1, N_A * sizeof(int));
    cudaMalloc(&d_out_taskA2, N_A * sizeof(int));
    
    // TaskB1, TaskB2: 10^9 elements each
    int* d_out_taskB1, *d_out_taskB2;
    int* h_out_taskB1 = (int*)malloc(N_B * sizeof(int));
    int* h_out_taskB2 = (int*)malloc(N_B * sizeof(int));
    
    cudaMalloc(&d_out_taskB1, N_B * sizeof(int));
    cudaMalloc(&d_out_taskB2, N_B * sizeof(int));
    
    // Task B setup
    int* d_block_sums1, *d_block_sums2;
    cudaMalloc(&d_block_sums1, num_blocks_B * sizeof(int));
    cudaMalloc(&d_block_sums2, num_blocks_B * sizeof(int));
    
    // Start total timing (including memory copy)
    cudaEventRecord(start_total);
    
    // Start kernel-only timing
    cudaEventRecord(start_kernel);
    
    // Launch Task A1 (single-block, 10^6 elements) on stream1
    cudaEventRecord(start_taskA1, stream1);
    task_a_single_block_scan<<<1, 1024, shared_mem_size_A, stream1>>>(d_in_A, d_out_taskA1, N_A);
    cudaEventRecord(stop_taskA1, stream1);
    
    // Launch Task A2 (single-block, 10^6 elements) on stream2
    cudaEventRecord(start_taskA2, stream2);
    task_a_single_block_scan<<<1, 1024, shared_mem_size_A, stream2>>>(d_in_A, d_out_taskA2, N_A);
    cudaEventRecord(stop_taskA2, stream2);
    
    // Launch Task B1 (two-phase, 10^9 elements) on stream3
    cudaEventRecord(start_taskB1, stream3);
    task_b_phase1_block_scan<<<num_blocks_B, 1024, shared_mem_size_B, stream3>>>(d_in_B, d_out_taskB1, d_block_sums1, N_B, elements_per_block);
    task_b_phase2_scan_block_sums<<<1, 1024, shared_mem_phase2_B, stream3>>>(d_block_sums1, num_blocks_B);
    task_b_phase3_add_block_prefix<<<num_blocks_B, 1024, 0, stream3>>>(d_out_taskB1, d_block_sums1, N_B, elements_per_block);
    cudaEventRecord(stop_taskB1, stream3);
    
    // Launch Task B2 (two-phase, 10^9 elements) on stream4
    cudaEventRecord(start_taskB2, stream4);
    task_b_phase1_block_scan<<<num_blocks_B, 1024, shared_mem_size_B, stream4>>>(d_in_B, d_out_taskB2, d_block_sums2, N_B, elements_per_block);
    task_b_phase2_scan_block_sums<<<1, 1024, shared_mem_phase2_B, stream4>>>(d_block_sums2, num_blocks_B);
    task_b_phase3_add_block_prefix<<<num_blocks_B, 1024, 0, stream4>>>(d_out_taskB2, d_block_sums2, N_B, elements_per_block);
    cudaEventRecord(stop_taskB2, stream4);
    
    // Wait for all kernels to complete
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel);
    
    // Copy results back using streams
    cudaMemcpyAsync(h_out_taskA1, d_out_taskA1, N_A * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_out_taskA2, d_out_taskA2, N_A * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(h_out_taskB1, d_out_taskB1, N_B * sizeof(int), cudaMemcpyDeviceToHost, stream3);
    cudaMemcpyAsync(h_out_taskB2, d_out_taskB2, N_B * sizeof(int), cudaMemcpyDeviceToHost, stream4);
    
    // Wait for all memory copies to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);
    
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    
    // Calculate and report timing results
    float time_total, time_kernel_only;
    float time_taskA1, time_taskA2, time_taskB1, time_taskB2;
    
    cudaEventElapsedTime(&time_total, start_total, stop_total);
    cudaEventElapsedTime(&time_kernel_only, start_kernel, stop_kernel);
    cudaEventElapsedTime(&time_taskA1, start_taskA1, stop_taskA1);
    cudaEventElapsedTime(&time_taskA2, start_taskA2, stop_taskA2);
    cudaEventElapsedTime(&time_taskB1, start_taskB1, stop_taskB1);
    cudaEventElapsedTime(&time_taskB2, start_taskB2, stop_taskB2);
    
    printf("\n=== Execution Time Report ===\n");
    printf("Task C Total time (with memory copy): %.3f ms\n", time_total);
    printf("Task C Kernel-only time (without memory copy): %.3f ms\n", time_kernel_only);
    printf("Task A1 (single-block) time: %.3f ms\n", time_taskA1);
    printf("Task A2 (single-block) time: %.3f ms\n", time_taskA2);
    printf("Task B1 (multi-block) time: %.3f ms\n", time_taskB1);
    printf("Task B2 (multi-block) time: %.3f ms\n", time_taskB2);
    
    printf("\n=== Performance Comparison Summary ===\n");
    printf("CPU Task A (10^6 elements):\n");
    printf("  - Total time: %.3f ms\n", time_cpu_A);
    printf("CPU Task B (10^9 elements):\n");
    printf("  - Total time: %.3f ms\n", time_cpu_B);
    printf("Task A (Single Block, 10^6 elements):\n");
    printf("  - Total time: %.3f ms\n", time_taskA_total);
    printf("  - Kernel-only time: %.3f ms\n", time_taskA_kernel);
    printf("Task B (Multi Block, 10^9 elements):\n");
    printf("  - Total time: %.3f ms\n", time_taskB_total);
    printf("  - Kernel-only time: %.3f ms\n", time_taskB_kernel);
    printf("Task C (Concurrent):\n");
    printf("  - Total time: %.3f ms\n", time_total);
    printf("  - Kernel-only time: %.3f ms\n", time_kernel_only);
    
    // Print first 20 elements for verification
    printf("\n=== Results Verification ===\n");
    printf("CPU Task A: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_cpu_A[i]);
    }
    printf("\nTask A1: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_taskA1[i]);
    }
    printf("\nTask A2: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_taskA2[i]);
    }
    printf("\nTask B1: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_taskB1[i]);
    }
    printf("\nTask B2: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_out_taskB2[i]);
    }
    printf("\n");
    
    // Cleanup streams and events
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_taskA1);
    cudaEventDestroy(stop_taskA1);
    cudaEventDestroy(start_taskA2);
    cudaEventDestroy(stop_taskA2);
    cudaEventDestroy(start_taskB1);
    cudaEventDestroy(stop_taskB1);
    cudaEventDestroy(start_taskB2);
    cudaEventDestroy(stop_taskB2);
    
    // Cleanup additional memory
    cudaFree(d_out_taskA1);
    cudaFree(d_out_taskA2);
    cudaFree(d_out_taskB1);
    cudaFree(d_out_taskB2);
    cudaFree(d_block_sums1);
    cudaFree(d_block_sums2);
    
    free(h_out_taskA1);
    free(h_out_taskA2);
    free(h_out_taskB1);
    free(h_out_taskB2);

    // Cleanup Task A memory
    cudaFree(d_in_A); 
    cudaFree(d_out_A);
    free(h_in_A);
    free(h_out_A);
    free(h_out_cpu_A);
    
    // Cleanup Task B memory
    cudaFree(d_in_B); 
    cudaFree(d_out_B);
    cudaFree(d_block_sums_B);
    free(h_in_B);
    free(h_out_taskB);
    free(h_out_cpu_B);
    return 0;
}