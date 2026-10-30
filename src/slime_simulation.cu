#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>

#define WIDTH 256
#define HEIGHT 256
#define PARTICLES 10000
#define ITERATIONS 250
#define PI 3.14159265358979323846f

struct Particle {
    float x;
    float y;
    float theta;
};

__device__ int wrapIndex(int v, int limit) {
    int r = v % limit;
    return (r < 0) ? r + limit : r;
}

__device__ float sense(const float* grid, float x, float y) {
    int ix = wrapIndex((int)floorf(x), WIDTH);
    int iy = wrapIndex((int)floorf(y), HEIGHT);
    return grid[iy * WIDTH + ix];
}

__global__ void moveParticles(
    Particle* particles,
    const float* oldGrid,
    float* depositGrid
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= PARTICLES) return;

    Particle p = particles[id];

    float sensorDistance = 9.0f;
    float sensorAngle = PI / 4.0f;
    float turnAngle = PI / 8.0f;
    float speed = 1.0f;

    float left = p.theta + sensorAngle;
    float front = p.theta;
    float right = p.theta - sensorAngle;

    float leftValue = sense(
        oldGrid,
        p.x + cosf(left) * sensorDistance,
        p.y + sinf(left) * sensorDistance
    );

    float frontValue = sense(
        oldGrid,
        p.x + cosf(front) * sensorDistance,
        p.y + sinf(front) * sensorDistance
    );

    float rightValue = sense(
        oldGrid,
        p.x + cosf(right) * sensorDistance,
        p.y + sinf(right) * sensorDistance
    );

    if (leftValue > frontValue && leftValue > rightValue) {
        p.theta += turnAngle;
    }
    else if (rightValue > frontValue && rightValue > leftValue) {
        p.theta -= turnAngle;
    }

    p.x = fmodf(
        p.x + cosf(p.theta) * speed + WIDTH,
        WIDTH
    );

    p.y = fmodf(
        p.y + sinf(p.theta) * speed + HEIGHT,
        HEIGHT
    );

    particles[id] = p;

    int gx = wrapIndex((int)floorf(p.x), WIDTH);
    int gy = wrapIndex((int)floorf(p.y), HEIGHT);

    atomicAdd(
        &depositGrid[gy * WIDTH + gx],
        0.05f
    );
}

__global__ void diffusePheromones(
    const float* oldGrid,
    float* newGrid,
    float evaporation
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    float sum = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = wrapIndex(x + dx, WIDTH);
            int ny = wrapIndex(y + dy, HEIGHT);

            sum += oldGrid[ny * WIDTH + nx];
        }
    }

    float diffused = (sum / 9.0f) * evaporation;

    int index = y * WIDTH + x;

    float value = diffused + newGrid[index];

    newGrid[index] = fminf(
        1.0f,
        fmaxf(0.0f, value)
    );
}

int main() {
    std::cout << "====================================\n";
    std::cout << "       LET'S MAKE SOME SLIME!\n";
    std::cout << "       CUDA PHYSARUM SIMULATION\n";
    std::cout << "====================================\n";

    std::vector<Particle> particles(PARTICLES);

    std::mt19937 rng(42);

    std::uniform_real_distribution<float> position(
        WIDTH * 0.4f,
        WIDTH * 0.6f
    );

    std::uniform_real_distribution<float> angle(
        0.0f,
        2.0f * PI
    );

    for (int i = 0; i < PARTICLES; i++) {
        particles[i].x = position(rng);
        particles[i].y = position(rng);
        particles[i].theta = angle(rng);
    }

    Particle* d_particles;
    float *gridA, *gridB;

    size_t particleBytes = PARTICLES * sizeof(Particle);
    size_t gridBytes = WIDTH * HEIGHT * sizeof(float);

    cudaMalloc(&d_particles, particleBytes);
    cudaMalloc(&gridA, gridBytes);
    cudaMalloc(&gridB, gridBytes);

    cudaMemcpy(
        d_particles,
        particles.data(),
        particleBytes,
        cudaMemcpyHostToDevice
    );

    cudaMemset(gridA, 0, gridBytes);
    cudaMemset(gridB, 0, gridBytes);

    int threads = 256;

    int particleBlocks =
        (PARTICLES + threads - 1) / threads;

    dim3 block2D(16, 16);

    dim3 grid2D(
        (WIDTH + 15) / 16,
        (HEIGHT + 15) / 16
    );

    float evaporation = 0.97f;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < ITERATIONS; step++) {
        cudaMemset(gridB, 0, gridBytes);

        moveParticles<<<particleBlocks, threads>>>(
            d_particles,
            gridA,
            gridB
        );

        diffusePheromones<<<grid2D, block2D>>>(
            gridA,
            gridB,
            evaporation
        );

        std::swap(gridA, gridB);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    std::vector<float> result(WIDTH * HEIGHT);

    cudaMemcpy(
        result.data(),
        gridA,
        gridBytes,
        cudaMemcpyDeviceToHost
    );

    float total = 0.0f;
    float maximum = 0.0f;

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        total += result[i];
        maximum = std::max(maximum, result[i]);
    }

    float average = total / (WIDTH * HEIGHT);

    std::cout << "\nSIMULATION COMPLETED!\n";
    std::cout << "Particles: " << PARTICLES << "\n";
    std::cout << "Grid: " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "Average pheromone: " << average << "\n";
    std::cout << "Maximum pheromone: " << maximum << "\n";
    std::cout << "GPU execution time: " << milliseconds << " ms\n";

    std::cout << "\n========== FINAL SLIME PATTERN ==========\n";

    float peak = *std::max_element(
        result.begin(),
        result.end()
    );

    if (peak < 0.000001f) peak = 1.0f;

    const char* shades = " .:-=+*#%@";

    for (int y = 0; y < HEIGHT; y += 4) {
        for (int x = 0; x < WIDTH; x += 2) {
            float v = result[y * WIDTH + x] / peak;

            int level = (int)(pow(v, 0.4f) * 9.0f);

            level = std::min(9, level);

            std::cout << shades[level];
        }

        std::cout << "\n";
    }

    std::cout << "=========================================\n";
    std::cout << "GPU memory released successfully.\n";

    cudaFree(d_particles);
    cudaFree(gridA);
    cudaFree(gridB);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
