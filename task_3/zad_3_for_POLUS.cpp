#include <mpi.h>
#include <iostream>
#include <vector>

void bad_access(int N, int rank, int size, MPI_Win win, int* shared_array) {
    int local_sum = 0;

    MPI_Win_fence(0, win);

    for (int i = rank; i < N; i += size) {
        int value;
        MPI_Get(&value, 1, MPI_INT, 0, i, 1, MPI_INT, win);
        local_sum += value;
    }

    MPI_Win_fence(0, win);
    MPI_Accumulate(&local_sum, 1, MPI_INT, 0, N, 1, MPI_INT, MPI_SUM, win);
    MPI_Win_fence(0, win);
}

void good_access(int N, int rank, int size, MPI_Win win, int* shared_array) {
    int block_size = N / size;
    int start = rank * block_size;
    int end = (rank == size - 1) ? N : start + block_size;
    int my_count = end - start;
    std::vector<int> local_block(my_count);

    MPI_Win_fence(0, win);
    MPI_Get(local_block.data(), my_count, MPI_INT, 0, start, my_count, MPI_INT, win);
    MPI_Win_fence(0, win);

    int local_sum = 0;
    for (int v : local_block)
        local_sum += v;

    MPI_Win_fence(0, win);
    MPI_Accumulate(&local_sum, 1, MPI_INT, 0, N, 1, MPI_INT, MPI_SUM, win);
    MPI_Win_fence(0, win);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 50000000;
    int* shared_array = nullptr;
    MPI_Win win;

    if (rank == 0) {
        std::cout << "N_processes: " << size << " ";
        shared_array = new int[N + 1];
        for (int i = 0; i < N; ++i)
            shared_array[i] = 1;
        shared_array[N] = 0;
    }

    MPI_Win_create(shared_array, (rank == 0 ? (N + 1) * sizeof(int) : 0), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    double start_time, end_time;

    if (rank == 0) shared_array[N] = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    bad_access(N, rank, size, win, shared_array);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0)
        std::cout << "bad_access: " << (end_time - start_time);

    if (rank == 0) shared_array[N] = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    good_access(N, rank, size, win, shared_array);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0)
        std::cout << "good_access: " << (end_time - start_time);

    if (rank == 0) delete[] shared_array;
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
