#include <iostream>
#include <vector>
#include <chrono>

const int N = 1024;

std::vector<std::vector<int>> create_matrix() {
    srand(0);
    std::vector<std::vector<int>> mat(N, std::vector<int> (N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = rand() % 97;
        }
    }
    return mat;
}

std::vector<std::vector<int>> gemm(std::vector<std::vector<int>> &a, std::vector<std::vector<int>> &b) {
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

void mem_copy(std::vector<std::vector<int>> &a, int *d_a) {
    int *temp = new int[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp[i * N + j] = a[i][j];
        }
    }
    cudaMemcpy(d_a, temp, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    delete []temp;
}
void check(int *d_a, std::vector<std::vector<int>> &a) {
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
    // Define tile size
    const int TILE_SIZE = 32;
    
    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for tiles
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];
    
    int sum = 0;
    
    // Number of tiles needed
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over tiles
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of A into shared memory
        int a_row = by * TILE_SIZE + ty;
        int a_col = tile * TILE_SIZE + tx;
        if (a_row < n && a_col < n) {
            As[ty][tx] = a[a_row * n + a_col];
        } else {
            As[ty][tx] = 0;
        }
        
        // Load tile of B into shared memory
        int b_row = tile * TILE_SIZE + ty;
        int b_col = bx * TILE_SIZE + tx;
        if (b_row < n && b_col < n) {
            Bs[ty][tx] = b[b_row * n + b_col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main() {

    auto a = create_matrix();
    auto b = create_matrix();
    
    auto cpu_gemm_start = std::chrono::high_resolution_clock::now();
    auto c = gemm(a, b);
    auto cpu_gemm_end = std::chrono::high_resolution_clock::now();

    int *d_a, *d_b, *d_c; // on device
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    mem_copy(a, d_a);
    mem_copy(b, d_b);

    auto simple_gpu_gemm_start = std::chrono::high_resolution_clock::now();
    // Use tiled kernel with 2D block and grid configuration
    dim3 blockSize(32, 32);  // 32x32 threads per block
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);  // Calculate number of blocks needed
    d_gemm_tiled<<<gridSize, blockSize>>> (d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    auto simple_gpu_gemm_end = std::chrono::high_resolution_clock::now();
    check(d_c, c);

    std::cout << "CPU GEMM Time (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds> (cpu_gemm_end - cpu_gemm_start).count() << std::endl;
    std::cout << "GPU GEMM Time (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds> (simple_gpu_gemm_end - simple_gpu_gemm_start).count() << std::endl;

    return 0;
}