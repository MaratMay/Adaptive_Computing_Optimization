#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define DEFAULT_COL_BLOCK 0

static double Aij(int i, int j, int n)
{
    return (i == j) ? (double)(n + 1) : 1.0;
}

static double x_exact(int i)
{
    (void)i;
    return 1.0;
}

typedef struct {
    int n;
    int rank;
    int size;
    int col_block;
    int local_cols;
    int *global_cols;
    double *A;
    double *b;
    double *b_orig;
} DistMatrix;

static int owner(int global_col, int col_block, int size)
{
    int block = global_col / col_block;
    return block % size;
}

static int local_idx(int global_col, int col_block, int rank, int size)
{
    if(owner(global_col, col_block, size) != rank)
        return -1;

    int block = global_col / col_block;
    int block_offset = (block / size) * col_block;
    int in_block = global_col % col_block;
    return block_offset + in_block;
}

static void dist_init(DistMatrix *d, int n, int col_block, int rank, int size)
{
    d->n = n;
    d->rank = rank;
    d->size = size;
    d->col_block = col_block;

    d->local_cols = 0;
    for(int block = rank; ; block += size) {
        int begin = block * col_block;
        if(begin >= n) break;
        int block_cols = (begin + col_block <= n) ? col_block : (n - begin);
        d->local_cols += block_cols;
    }

    d->global_cols = malloc(d->local_cols * sizeof(int));
    int idx = 0;
    for(int block = rank; ; block += size) {
        int begin = block * col_block;
        if(begin >= n) break;
        int block_cols = (begin + col_block <= n) ? col_block : (n - begin);
        for(int offset = 0; offset < block_cols; offset++) {
            d->global_cols[idx++] = begin + offset;
        }
    }

    d->A = calloc((size_t)d->local_cols * (size_t)n, sizeof(double));
    d->b = calloc((size_t)n, sizeof(double));
    d->b_orig = calloc((size_t)n, sizeof(double));
}

static void dist_free(DistMatrix *d)
{
    free(d->global_cols);
    free(d->A);
    free(d->b);
    free(d->b_orig);
}

static double* dist_col(DistMatrix *d, int local_col)
{
    return &d->A[(size_t)local_col * (size_t)d->n];
}

static void dist_generate(DistMatrix *d)
{
    double *b_local = calloc((size_t)d->n, sizeof(double));

    for(int lcol = 0; lcol < d->local_cols; lcol++) {
        int gcol = d->global_cols[lcol];
        double xj = x_exact(gcol);
        double *col = dist_col(d, lcol);

        for(int i = 0; i < d->n; i++) {
            double val = Aij(i, gcol, d->n);
            col[i] = val;
            b_local[i] += val * xj;
        }
    }

    MPI_Allreduce(b_local, d->b, d->n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(d->b_orig, d->b, (size_t)d->n * sizeof(double));

    free(b_local);
}

static void householder_step(DistMatrix *d, int k, double *reflector)
{
    int own = owner(k, d->col_block, d->size);
    int tail = d->n - k;

    if(d->rank == own) {
        int lcol = local_idx(k, d->col_block, d->rank, d->size);
        double *kcol = dist_col(d, lcol);

        double norm = 0.0;
        for(int i = k; i < d->n; i++) {
            norm += kcol[i] * kcol[i];
        }
        norm = sqrt(norm);

        reflector[0] = norm;

        if(norm != 0.0) {
            for(int i = 0; i < tail; i++) {
                reflector[1 + i] = kcol[k + i] - ((i == 0) ? norm : 0.0);
            }

            double u_norm2 = 2.0 * norm * (norm - kcol[k]);
            if(u_norm2 < 0) u_norm2 = 0;
            double u_norm = sqrt(u_norm2);

            if(u_norm != 0.0) {
                double inv_norm = 1.0 / u_norm;
                for(int i = 0; i < tail; i++) {
                    reflector[1 + i] *= inv_norm;
                }
                reflector[0] = inv_norm;
            } else {
                reflector[0] = 0.0;
            }

            kcol[k] = norm;
            for(int i = 1; i < tail; i++) {
                kcol[k + i] = 0.0;
            }
        } else {
            reflector[0] = 0.0;
        }
    }

    MPI_Bcast(reflector, 1 + tail, MPI_DOUBLE, own, MPI_COMM_WORLD);

    double beta = reflector[0];
    if(beta == 0.0) return;

    double *u = &reflector[1];

    for(int lcol = 0; lcol < d->local_cols; lcol++) {
        int gcol = d->global_cols[lcol];
        if(gcol <= k) continue;

        double *col = dist_col(d, lcol);
        double dot = 0.0;
        for(int i = 0; i < tail; i++) {
            dot += u[i] * col[k + i];
        }
        double gamma = 2.0 * dot;
        for(int i = 0; i < tail; i++) {
            col[k + i] -= gamma * u[i];
        }
    }

    double dot_b = 0.0;
    for(int i = 0; i < tail; i++) {
        dot_b += u[i] * d->b[k + i];
    }
    double gamma_b = 2.0 * dot_b;
    for(int i = 0; i < tail; i++) {
        d->b[k + i] -= gamma_b * u[i];
    }
}

static double forward_elimination(DistMatrix *d)
{
    double *reflector = malloc((size_t)d->n * sizeof(double));

    double start = MPI_Wtime();

    for(int k = 0; k < d->n - 1; k++) {
        householder_step(d, k, reflector);
    }

    double end = MPI_Wtime();
    free(reflector);

    double local_time = end - start;
    double max_time;
    MPI_Allreduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return max_time;
}

static double back_substitution(DistMatrix *d, double *x)
{
    double start = MPI_Wtime();

    int block_count = (d->n + d->col_block - 1) / d->col_block;
    double *known_local = malloc((size_t)d->col_block * sizeof(double));
    double *known_global = malloc((size_t)d->col_block * sizeof(double));

    for(int i = 0; i < d->n; i++) x[i] = 0.0;

    for(int block = block_count - 1; block >= 0; block--) {
        int block_begin = block * d->col_block;
        int block_end = block_begin + d->col_block - 1;
        if(block_end >= d->n) block_end = d->n - 1;
        int block_size = block_end - block_begin + 1;
        int owner_rank = block % d->size;

        for(int i = 0; i < block_size; i++) known_local[i] = 0.0;

        for(int lcol = 0; lcol < d->local_cols; lcol++) {
            int gcol = d->global_cols[lcol];
            if(gcol <= block_end) continue;

            double *col = dist_col(d, lcol);
            double xj = x[gcol];

            for(int i = 0; i < block_size; i++) {
                int row = block_begin + i;
                known_local[i] += col[row] * xj;
            }
        }

        MPI_Reduce(known_local, known_global, block_size, MPI_DOUBLE,
                   MPI_SUM, owner_rank, MPI_COMM_WORLD);

        if(d->rank == owner_rank) {
            int first_local = local_idx(block_begin, d->col_block, d->rank, d->size);

            for(int i = block_end; i >= block_begin; i--) {
                int row_in_block = i - block_begin;
                double sum = known_global[row_in_block];

                for(int j = i + 1; j <= block_end; j++) {
                    int col_local = first_local + (j - block_begin);
                    double *col = dist_col(d, col_local);
                    sum += col[i] * x[j];
                }

                int col_local = first_local + row_in_block;
                double *col = dist_col(d, col_local);
                x[i] = (d->b[i] - sum) / col[i];
            }
        }

        MPI_Bcast(&x[block_begin], block_size, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);
    }

    free(known_local);
    free(known_global);

    double end = MPI_Wtime();
    double local_time = end - start;
    double max_time;
    MPI_Allreduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return max_time;
}

static double residual_norm(DistMatrix *d, double *x)
{
    int row_begin = (d->rank * d->n) / d->size;
    int row_end = ((d->rank + 1) * d->n) / d->size;

    double local_norm = 0.0;

    for(int i = row_begin; i < row_end; i++) {
        double Ax = 0.0;
        for(int j = 0; j < d->n; j++) {
            Ax += Aij(i, j, d->n) * x[j];
        }
        double r = Ax - d->b_orig[i];
        local_norm += r * r;
    }

    double global_norm;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_norm);
}

static double solution_error(DistMatrix *d, double *x)
{
    int row_begin = (d->rank * d->n) / d->size;
    int row_end = ((d->rank + 1) * d->n) / d->size;

    double local_err = 0.0;
    for(int i = row_begin; i < row_end; i++) {
        double diff = x[i] - x_exact(i);
        local_err += diff * diff;
    }

    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_err);
}

static int choose_block_size(int n, int size, int user_block)
{
    if(user_block > 0) {
        return (user_block < n) ? user_block : n;
    }

    int target_blocks = size * 8;
    int block = (n + target_blocks - 1) / target_blocks;

    if(block < 16) block = 16;
    if(block > 128) block = 128;
    if(block > n) block = n;

    return block;
}

static void run_test(int n, int rank, int size)
{
    int col_block = choose_block_size(n, size, DEFAULT_COL_BLOCK);

    if(rank == 0) {
        printf("Matrix size: n = %d\n", n);
        printf("Matrix type: Diagonally dominant (a_ii = n+1, a_ij = 1)\n");
        printf("Processes: %d\n", size);
        printf("Column block size: %d\n", col_block);
        printf("========================================\n\n");
    }

    DistMatrix d;
    dist_init(&d, n, col_block, rank, size);
    dist_generate(&d);

    double T1 = forward_elimination(&d);

    double *x = malloc((size_t)n * sizeof(double));
    double T2 = back_substitution(&d, x);

    double residual = residual_norm(&d, x);
    double error = solution_error(&d, x);

    if(rank == 0) {
        printf("RESULTS n = %d:\n", n);
        printf("  T1 (forward)  = %.6f sec\n", T1);
        printf("  T2 (backward) = %.6f sec\n", T2);
        printf("  Total         = %.6f sec\n", T1 + T2);
        printf("  ||Ax - b||    = %e\n", residual);
        printf("  ||x - x*||    = %e\n", error);
        printf("========================================\n");
    }

    free(x);
    dist_free(&d);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_list[] = {2000, 4000, 6000, 8000};
    int num_tests = 4;

    for(int t = 0; t < num_tests; t++) {
        run_test(n_list[t], rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}