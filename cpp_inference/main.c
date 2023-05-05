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
            test_correct++;
        }
        else {
            test_incorrect++;
        }
    }

    printf("Total tests: %d\n", total_tests);
    printf("Correct tests: %d\n", test_correct);
    printf("Accuracy: %f\n", (float)test_correct / (float)total_tests);

    return 0;
}
