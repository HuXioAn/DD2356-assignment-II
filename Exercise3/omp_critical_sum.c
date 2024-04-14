#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double omp_critical_sum(double *x, size_t size)
{
    double sum_val = 0.0;

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
#pragma omp critical
        {
            sum_val += x[i];
        }
    }

    return sum_val;
}

void generate_random(double *input, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        input[i] = rand() / (double)RAND_MAX;
    }
}

int main(int argc, char *argv[])
{

    int num_runs = atoi(argv[1]);

    size_t size = 10000000;
    double *array = (double *)malloc(size * sizeof(double));

    generate_random(array, size);

    int thread_counts[] = {1, 2, 4, 8, 16, 20, 24, 28, 32};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int j = 0; j < num_thread_counts; ++j)
    {
        int num_threads = thread_counts[j];
        double total_time = 0.0;

        for (int i = 0; i < num_runs; ++i)
        {
            double start_time = omp_get_wtime();
            omp_set_num_threads(num_threads);
            omp_critical_sum(array, size);
            double end_time = omp_get_wtime();
            total_time += (end_time - start_time);
        }

        double average_time = total_time / num_runs;
        printf("Threads: %d, Average Time: %.6f seconds\n", num_threads, average_time);
    }

    free(array);

    return 0;
}