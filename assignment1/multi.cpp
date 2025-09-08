#include <iostream>
#include <thread>
#include <mutex>

int total = 0;
const int N = 1000;

int matrices[4][N][N];
std::mutex mtx;

void mul(int id0, int id1) {
    int sum[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum[i][j] = 0;
    
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                sum[i][j] += matrices[id0][i][k] * matrices[id1][k][j];
            
    mtx.lock();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total = (total + sum[i][j]) % 10000000;
    mtx.unlock();
}

void initData() {
    srand(0);
    for (int t = 0; t <= 3; t++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrices[t][i][j] = rand() % 97;
            }
        }
    }
}

int main() {
    initData();

    std::thread t0(mul, 0, 1);
    std::thread t1(mul, 1, 2);
    std::thread t2(mul, 2, 3);
    std::thread t3(mul, 1, 3);
    t0.join();
    t1.join();
    t2.join();
    t3.join();

    std::cout << total << std::endl;
    return 0;
}