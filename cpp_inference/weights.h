#include "nnom.h"

#define DENSE_KERNEL_0 {70, 30, 71, -38, -12, 51, -3, 15, 85, 7, 51, -31, 2, 32, 17, 27, -32, 30, 62, -22, 5, 2, 11, -66, 48, 52, -16, -33, -28, 20, 52, 24, 56, 48, 26, -39, -6, -70, -55, 55, 70, 3, 41, 25, -33, 12, -31, -41, -59, 17, 13, -23, 93, -62, -6, 47, -62, -31, -54, 28, 34, -72, 8, -31, -62, -28, 41, 51, 62, 31, 30, -59, -36, 62, -4, 21, -3, -26, 24, 9, 42, 51, 41, -18, 27, -58, 77, -70, -18, 59, 3, 23, 20, -44, 90, -58, 20, 8, -32, -15, -42, 26, -25, -25, -37, 36, 56, -24, 60, 6, 1, -40, -17, -31, -44, 58, -56, -66, -46, -3, 34, 36, -1, -40, 25, 57, 75, 9, -35, 43, 15, -39, -32, -5, 33, -46, 51, -45, -42, -41, -31, -26, -9, 19}

#define DENSE_KERNEL_0_SHIFT (7)

#define DENSE_BIAS_0 {24, -80, 49, 13, 14, -47, 36, -21, 98, 65, 73, 74}

#define DENSE_BIAS_0_SHIFT (7)


#define DENSE_1_KERNEL_0 {33, -1, 41, -16, 11, 24, 12, 30, -32, -21, 8, 33, -35, 6, -10, -16, 31, 6, -21, -16, -21, -28, 10, 26, -16, 33, 8, 28, -12, 28, 17, -13, 7, 34, 21, -7, 19, -9, 55, 42, -17, -18, 17, -50, 27, 34, 29, 57, 23, -25, -17, 9, 14, -1, 59, 28, 1, 25, 11, 8, -7, -38, -17, 0, -39, -42, 34, 5, -14, -13, 23, 16, 20, 36, 8, -23, -50, -20, -4, -5, 39, -66, 9, -87, 43, 4, 26, 8, 16, -57, -64, -9, 24, -37, 47, -34}

#define DENSE_1_KERNEL_0_SHIFT (6)

#define DENSE_1_BIAS_0 {-10, 10, -123, 114, 11, 21, -7, -49}

#define DENSE_1_BIAS_0_SHIFT (10)


#define DENSE_2_KERNEL_0 {27, 8, -3, 24, -10, -36, -24, -25, -34, -51, -30, 27, 4, 32, 25, 10, -37, 1, 19, 73, 49, 34, -71, -47, 6, 89, 18, -36, -75, -85, 23, 0, -9, 26, 5, 34, -46, -5, -46, -21, -28, 36, 23, -28, 33, -18, 32, 14, 6, 5, 16, 15, 0, -25, -48, -37, 58, -1, 30, -5, -52, 19, -6, -19}

#define DENSE_2_KERNEL_0_SHIFT (6)

#define DENSE_2_BIAS_0 {0, -8, 51, 44, -49, -29, 86, -42}

#define DENSE_2_BIAS_0_SHIFT (9)


#define DENSE_3_KERNEL_0 {63, -62, 106, 118, -70, -49, 31, -13, 27, 98, 3, 33, 44, 45, -81, -73}

#define DENSE_3_KERNEL_0_SHIFT (7)

#define DENSE_3_BIAS_0 {89, -89}

#define DENSE_3_BIAS_0_SHIFT (10)



/* output enconding for each layer */
#define DENSE_INPUT_OUTPUT_SHIFT 0
#define DENSE_OUTPUT_SHIFT -2
#define ACTIVATION_OUTPUT_SHIFT 7
#define DENSE_1_OUTPUT_SHIFT 5
#define ACTIVATION_1_OUTPUT_SHIFT 5
#define DENSE_2_OUTPUT_SHIFT 5
#define ACTIVATION_2_OUTPUT_SHIFT 5
#define DENSE_3_OUTPUT_SHIFT 5
#define ACTIVATION_3_OUTPUT_SHIFT 7

/* bias shift and output shift for each layer */
#define DENSE_OUTPUT_RSHIFT (DENSE_INPUT_OUTPUT_SHIFT+DENSE_KERNEL_0_SHIFT-DENSE_OUTPUT_SHIFT)
#define DENSE_BIAS_LSHIFT   (DENSE_INPUT_OUTPUT_SHIFT+DENSE_KERNEL_0_SHIFT-DENSE_BIAS_0_SHIFT)
#if DENSE_OUTPUT_RSHIFT < 0
#error DENSE_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_BIAS_LSHIFT < 0
#error DENSE_BIAS_RSHIFT must be bigger than 0
#endif
#define DENSE_1_OUTPUT_RSHIFT (ACTIVATION_OUTPUT_SHIFT+DENSE_1_KERNEL_0_SHIFT-DENSE_1_OUTPUT_SHIFT)
#define DENSE_1_BIAS_LSHIFT   (ACTIVATION_OUTPUT_SHIFT+DENSE_1_KERNEL_0_SHIFT-DENSE_1_BIAS_0_SHIFT)
#if DENSE_1_OUTPUT_RSHIFT < 0
#error DENSE_1_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_1_BIAS_LSHIFT < 0
#error DENSE_1_BIAS_RSHIFT must be bigger than 0
#endif
#define DENSE_2_OUTPUT_RSHIFT (ACTIVATION_1_OUTPUT_SHIFT+DENSE_2_KERNEL_0_SHIFT-DENSE_2_OUTPUT_SHIFT)
#define DENSE_2_BIAS_LSHIFT   (ACTIVATION_1_OUTPUT_SHIFT+DENSE_2_KERNEL_0_SHIFT-DENSE_2_BIAS_0_SHIFT)
#if DENSE_2_OUTPUT_RSHIFT < 0
#error DENSE_2_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_2_BIAS_LSHIFT < 0
#error DENSE_2_BIAS_RSHIFT must be bigger than 0
#endif
#define DENSE_3_OUTPUT_RSHIFT (ACTIVATION_2_OUTPUT_SHIFT+DENSE_3_KERNEL_0_SHIFT-DENSE_3_OUTPUT_SHIFT)
#define DENSE_3_BIAS_LSHIFT   (ACTIVATION_2_OUTPUT_SHIFT+DENSE_3_KERNEL_0_SHIFT-DENSE_3_BIAS_0_SHIFT)
#if DENSE_3_OUTPUT_RSHIFT < 0
#error DENSE_3_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_3_BIAS_LSHIFT < 0
#error DENSE_3_BIAS_RSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t dense_weights[] = DENSE_KERNEL_0;
static const nnom_weight_t dense_w = { (const void*)dense_weights, DENSE_OUTPUT_RSHIFT};
static const int8_t dense_bias[] = DENSE_BIAS_0;
static const nnom_bias_t dense_b = { (const void*)dense_bias, DENSE_BIAS_LSHIFT};
static const int8_t dense_1_weights[] = DENSE_1_KERNEL_0;
static const nnom_weight_t dense_1_w = { (const void*)dense_1_weights, DENSE_1_OUTPUT_RSHIFT};
static const int8_t dense_1_bias[] = DENSE_1_BIAS_0;
static const nnom_bias_t dense_1_b = { (const void*)dense_1_bias, DENSE_1_BIAS_LSHIFT};
static const int8_t dense_2_weights[] = DENSE_2_KERNEL_0;
static const nnom_weight_t dense_2_w = { (const void*)dense_2_weights, DENSE_2_OUTPUT_RSHIFT};
static const int8_t dense_2_bias[] = DENSE_2_BIAS_0;
static const nnom_bias_t dense_2_b = { (const void*)dense_2_bias, DENSE_2_BIAS_LSHIFT};
static const int8_t dense_3_weights[] = DENSE_3_KERNEL_0;
static const nnom_weight_t dense_3_w = { (const void*)dense_3_weights, DENSE_3_OUTPUT_RSHIFT};
static const int8_t dense_3_bias[] = DENSE_3_BIAS_0;
static const nnom_bias_t dense_3_b = { (const void*)dense_3_bias, DENSE_3_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[12];
static int8_t nnom_output_data[2];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[10];

	new_model(&model);

	layer[0] = Input(shape(12,1,1), nnom_input_data);
	layer[1] = model.hook(Dense(12, &dense_w, &dense_b), layer[0]);
	layer[2] = model.active(act_sigmoid(DENSE_OUTPUT_SHIFT), layer[1]);
	layer[3] = model.hook(Dense(8, &dense_1_w, &dense_1_b), layer[2]);
	layer[4] = model.active(act_relu(), layer[3]);
	layer[5] = model.hook(Dense(8, &dense_2_w, &dense_2_b), layer[4]);
	layer[6] = model.active(act_relu(), layer[5]);
	layer[7] = model.hook(Dense(2, &dense_3_w, &dense_3_b), layer[6]);
	layer[8] = model.hook(Softmax(), layer[7]);
	layer[9] = model.hook(Output(shape(2,1,1), nnom_output_data), layer[8]);
	model_compile(&model, layer[0], layer[9]);
	return &model;
}
