#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define N 5000

void generate_system(double *A, double *b, double *x_true, int n)
{
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < n; i++)
            A[i + j*n] = (double)rand()/RAND_MAX * 2.0 - 1.0;
    }
    
    for(int i = 0; i < n; i++)
        x_true[i] = 1.0;
    
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for(int j = 0; j < n; j++)
            sum += A[i + j*n] * x_true[j];
        b[i] = sum;
    }
}

void mirror_method(double *A, double *b, int n)
{
    double *u = malloc(n * sizeof(double));
    
    #pragma omp parallel shared(A, b, u, n)
    {
        for(int k = 0; k < n-1; k++)
        {
            #pragma omp single
            {
                int m = n - k;
                double norm_x = 0.0;
                
                for(int i = 0; i < m; i++)
                {
                    double val = A[(k+i) + k*n];
                    norm_x += val * val;
                }
                norm_x = sqrt(norm_x);
                
                if(norm_x != 0.0)
                {
                    double a_kk = A[k + k*n];
                    double alpha = (a_kk >= 0) ? -norm_x : norm_x;
                    
                    u[0] = a_kk - alpha;
                    for(int i = 1; i < m; i++)
                        u[i] = A[(k+i) + k*n];
                    
                    double u_norm = sqrt(2.0 * norm_x * (norm_x + fabs(a_kk)));
                    
                    if(u_norm != 0.0)
                    {
                        double inv = 1.0 / u_norm;
                        for(int i = 0; i < m; i++)
                            u[i] *= inv;
                        
                        A[k + k*n] = alpha;
                        
                        for(int i = 1; i < m; i++)
                            A[(k+i) + k*n] = 0.0;
                    }
                }
            }
            
            #pragma omp barrier
            
            if(u[0] != 0.0 || u[0] != 0.0)
            {
                #pragma omp for schedule(dynamic, 64)
                for(int j = k+1; j < n; j++)
                {
                    double dot = 0.0;
                    for(int i = 0; i < n-k; i++)
                        dot += u[i] * A[(k+i) + j*n];
                    
                    double tau = 2.0 * dot;
                    
                    for(int i = 0; i < n-k; i++)
                        A[(k+i) + j*n] -= tau * u[i];
                }
                
                #pragma omp single
                {
                    double dot_b = 0.0;
                    for(int i = 0; i < n-k; i++)
                        dot_b += u[i] * b[k+i];
                    
                    double tau_b = 2.0 * dot_b;
                    
                    for(int i = 0; i < n-k; i++)
                        b[k+i] -= tau_b * u[i];
                }
            }
            
            #pragma omp barrier
        }
    }
    
    free(u);
}

void back_sub(double *A, double *b, double *x, int n)
{
    for(int i = n-1; i >= 0; i--)
    {
        double sum = 0.0;
        for(int j = i+1; j < n; j++)
            sum += A[i + j*n] * x[j];
        x[i] = (b[i] - sum) / A[i + i*n];
    }
}

double residual(double *A, double *b, double *x, int n)
{
    double norm = 0.0;
    
    #pragma omp parallel for reduction(+:norm)
    for(int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for(int j = 0; j < n; j++)
            sum += A[i + j*n] * x[j];
        double d = sum - b[i];
        norm += d * d;
    }
    
    return sqrt(norm);
}

int main()
{
    int thread_counts[] = {1, 2, 4, 8, 12, 16, 24, 32};
    int num_tests = 8;
    int n = N;
    
    FILE *f = fopen("results.txt", "w");
    
    double *A = malloc(n * n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *x_true = malloc(n * sizeof(double));
    
    double *A_copy = malloc(n * n * sizeof(double));
    double *b_copy = malloc(n * sizeof(double));
    
    srand(0);
    
    for(int t = 0; t < num_tests; t++)
    {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);
        
        generate_system(A, b, x_true, n);
        
        memcpy(A_copy, A, n * n * sizeof(double));
        memcpy(b_copy, b, n * sizeof(double));
        
        double t1_start = omp_get_wtime();
        mirror_method(A, b, n);
        double t1_end = omp_get_wtime();
        
        double t2_start = omp_get_wtime();
        back_sub(A, b, x, n);
        double t2_end = omp_get_wtime();
        
        double T1 = t1_end - t1_start;
        double T2 = t2_end - t2_start;
        double Tall = T1 + T2;
        
        double r = residual(A_copy, b_copy, x, n);
        
        double diff = 0.0;
        for(int i = 0; i < n; i++)
        {
            double d = x[i] - x_true[i];
            diff += d * d;
        }
        diff = sqrt(diff);
        
        printf("threads=%d T1=%.6f T2=%.6f Tall=%.6f residual=%e error=%e\n",
               threads, T1, T2, Tall, r, diff);
        
        fprintf(f, "threads=%d T1=%.6f T2=%.6f Tall=%.6f residual=%e error=%e\n",
                threads, T1, T2, Tall, r, diff);
        
        fflush(f);
    }
    
    fclose(f);
    
    free(A);
    free(b);
    free(x);
    free(x_true);
    free(A_copy);
    free(b_copy);
    
    return 0;
}