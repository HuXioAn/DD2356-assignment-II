#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double omp_sum(double *x, size_t size)
{
    double sum_val = 0.0;

#pragma omp parallel for reduction(+ : sum_val)
    for (size_t i = 0; i < size; ++i)
    {
        sum_val += x[i];
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
    double *times = (double *)malloc(num_runs * sizeof(double));
    double total_time = 0.0;

    generate_random(array, size);

    for (int i = 0; i < num_runs; ++i)
    {
        double start_time = omp_get_wtime();
        omp_set_num_threads(32);
        omp_sum(array, size);
        double end_time = omp_get_wtime();
        times[i] = end_time - start_time;
        total_time += times[i];
    }

    double average_time = total_time / num_runs;

    double sum_sq_diff = 0.0;
    for (int i = 0; i < num_runs; ++i)
    {
        sum_sq_diff += (times[i] - average_time) * (times[i] - average_time);
    }
    double standard_deviation = sqrt(sum_sq_diff / num_runs);

    printf("Average Time: %.6f seconds\n", average_time);
    printf("Standard Deviation: %.6f seconds\n", standard_deviation);

    free(times);
    free(array);

    return 0;
}