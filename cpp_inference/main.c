#include <stdio.h>
#include "nnom.h"
#include "weights.h"
#include "test_data.h"
#include "test_labels.h"

int main()
{
    nnom_model_t *model;
    model = nnom_model_create();
    model_run(model);

    int total_tests = 0;
    int test_correct = 0;
    int test_incorrect = 0;

    int predicted_label;
    float probability;

    for (int i = 0; i < samples; ++i) {
        total_tests++;
        for (int j = 0; j < size_per_sample; ++j) {
            nnom_input_data[j] = (test_data[i][j]);
        }
        nnom_predict(model, &predicted_label, &probability);

        if (predicted_label == test_data_labels[i]){
            results[i] = 1;
            test_correct++;
        }
        else {
            results[i] = 0;
            test_incorrect++;
        }
    }

    printf("Total tests: %d\n", total_tests);
    printf("Correct tests: %d\n", test_correct);
    printf("Accuracy: %f\n", (float)test_correct / (float)total_tests);

    // Write each element of results to its own line in a file "results.csv" (creates file if it doesn't exists)
    FILE *fp;
    fp = fopen("./workspace/c/results.csv", "w+");
    for (int i = 0; i < samples; ++i) {
        fprintf(fp, "%d\n", results[i]);
    }
    fclose(fp);

    printf("Results written to results.csv\n");

    return 0;
}
