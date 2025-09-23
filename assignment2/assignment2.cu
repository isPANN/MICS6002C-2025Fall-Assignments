#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

std::vector<std::vector<int>> create_matrix(int N) {
    srand(0);
    std::vector<std::vector<int>> mat(N, std::vector<int> (N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = rand() % 97;
        }
    }
    return mat;
}

std::vector<std::vector<int>> gemm(std::vector<std::vector<int>> &a, std::vector<std::vector<int>> &b, int N) {
    std::vector<std::vector<int>> mat(N, std::vector<int> (N, 0));
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                mat[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return mat;
}

void mem_copy(std::vector<std::vector<int>> &a, int *d_a, int N) {
    int *temp = new int[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp[i * N + j] = a[i][j];
        }
    }
    cudaMemcpy(d_a, temp, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    delete []temp;
}

void check(int *d_a, std::vector<std::vector<int>> &a, int N) {
    bool equal = true;
    int *temp = new int[N * N];
    cudaMemcpy(temp, d_a, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (a[i][j] != temp[i * N + j]) {
                equal = false;
            }
        }
    }
    delete []temp;
    std::cout << "Check "<< (equal ? "Suceeded" : "Failed") << std::endl;
}

__global__ void d_gemm(int *a, int *b, int *c, int n) {
    int row = blockIdx.x, col = threadIdx.x;
    c[row * n + col] = 0;
    for (int i = 0; i < n; i++) {// a[row][i] * b[i][col]
        c[row * n + col] += a[row * n + i] * b[i * n + col];
    }
}

__global__ void d_gemm2(int *a, int *b, int *c, int n) {
    int row = blockIdx.x, col = threadIdx.x;
    int sum = 0;
    for (int i = 0; i < n; i++) {// a[row][i] * b[i][col]
        sum += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}

__global__ void d_gemm_tiled(int *a, int *b, int *c, int n) {
    const int TILE_SIZE = 16;
    
    int tx = threadIdx.x; // column within tile
    int ty = threadIdx.y; // row within tile
    
    int bx = blockIdx.x; // block column
    int by = blockIdx.y; // block row
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory tiles for A and B matrices
    __shared__ int tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_B[TILE_SIZE][TILE_SIZE];
    
    int result = 0;
    
    // Calculate how many tiles we need to process
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop through all tiles
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from matrix A into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < n && a_col < n) {
            tile_A[ty][tx] = a[a_row * n + a_col];
        } else {
            tile_A[ty][tx] = 0; // Padding with zeros
        }
        
        // Load tile from matrix B into shared memory
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < n && b_col < n) {
            tile_B[ty][tx] = b[b_row * n + b_col];
        } else {
            tile_B[ty][tx] = 0; // Padding with zeros
        }
        
        // Wait for all threads to finish loading
        __syncthreads();
        
        // Compute partial dot product using loaded tiles
        for (int k = 0; k < TILE_SIZE; k++) {
            result += tile_A[ty][k] * tile_B[k][tx];
        }
        
        // Wait before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < n && col < n) {
        c[row * n + col] = result;
    }
}

void test_matrix_size(int N) {
    std::cout << "\n=== Testing Matrix Size: " << N << "x" << N << " ===" << std::endl;
    
    auto a = create_matrix(N);
    auto b = create_matrix(N);
    
    // CPU matrix multiplication for verification
    auto cpu_gemm_start = std::chrono::high_resolution_clock::now();
    auto c = gemm(a, b, N);
    auto cpu_gemm_end = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));
    
    // Copy data to GPU
    mem_copy(a, d_a, N);
    mem_copy(b, d_b, N);
    
    // GPU matrix multiplication using simple tiled kernel
    auto gpu_gemm_start = std::chrono::high_resolution_clock::now();
    
    // Configure grid and block dimensions for simple tiled kernel
    const int TILE_SIZE = 16; // Same as in the kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE); // 16x16 threads per block
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    d_gemm_tiled<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    auto gpu_gemm_end = std::chrono::high_resolution_clock::now();
    
    // Verify correctness
    check(d_c, c, N);
    
    // Calculate and display timing results
    auto cpu_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_gemm_end - cpu_gemm_start).count();
    auto gpu_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_gemm_end - gpu_gemm_start).count();
    
    std::cout << "CPU GEMM Time: " << cpu_time_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "GPU GEMM Time: " << gpu_time_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Speedup: " << (double)cpu_time_ns / gpu_time_ns << "x" << std::endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    std::cout << "Matrix Multiplication with Simple Loop Tiling" << std::endl;
    std::cout << "=============================================" << std::endl;

    std::cout << "Task 2: Test size 1024x1024" << std::endl;
    test_matrix_size(1024);
    
    std::cout << "Task 3: Test different matrix sizes" << std::endl;
    std::vector<int> sizes = {128, 256, 512, 1024};
    
    std::cout << "\n| Matrix Size | GPU Time (ms) | CPU Time (ms) | Speedup |" << std::endl;
    std::cout << "|-------------|---------------|---------------|---------|" << std::endl;
    
    for (int N : sizes) {
        auto a = create_matrix(N);
        auto b = create_matrix(N);
        
        // CPU matrix multiplication
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto c = gemm(a, b, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        // Allocate GPU memory
        int *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, N * N * sizeof(int));
        cudaMalloc(&d_b, N * N * sizeof(int));
        cudaMalloc(&d_c, N * N * sizeof(int));
        
        // Copy data to GPU
        mem_copy(a, d_a, N);
        mem_copy(b, d_b, N);
        
        // GPU matrix multiplication
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        const int TILE_SIZE = 16;
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        
        d_gemm_tiled<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        
        check(d_c, c, N);
        
        auto cpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
        auto gpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;
        double speedup = cpu_time_ms / gpu_time_ms;
        
        std::cout << "| " << N << "x" << N << "     | " << std::fixed << std::setprecision(2) 
                  << gpu_time_ms << "        | " << cpu_time_ms << "        | " 
                  << speedup << "x    |" << std::endl;
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    
    std::cout << "\nNote: Using simple tiled kernel with 16x16 tile size" << std::endl;
    
    return 0;
}