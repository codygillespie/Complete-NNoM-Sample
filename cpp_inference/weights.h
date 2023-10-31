#include "nnom.h"

#define DENSE_KERNEL_0 {63, -23, 52, -43, -48, -15, -29, -32, 60, -28, 47, -37, -30, -18, -15, -32, 7, -1, 58, -40, -18, -40, -22, -68, 50, 2, 18, -44, -37, -27, -8, -20, 58, 1, 46, -52, -40, -74, -65, -25, 67, -26, 56, -23, -55, -39, -54, -83, -74, -32, -33, -47, 75, -59, 17, 3, -69, -53, -71, -23, 36, -56, 30, -39, -71, -49, -14, -18, 58, -7, 46, -53, -54, -11, -35, -30, 29, -37, 45, -19, -18, -12, -28, -50, 50, -56, 76, -67, -53, -10, -56, -33, 48, -52, 84, -65, 28, 56, -11, 40, 16, -36, 25, -42, -10, 61, 39, 46, 62, -40, 51, -30, 3, 40, -6, 80, 20, -41, 17, -31, 33, 72, 12, 27, 60, 9, 72, -28, -5, 72, 24, 43, 23, -28, 76, -36, 42, 36, -6, 47, 35, -33, 64, -2}

#define DENSE_KERNEL_0_SHIFT (6)

#define DENSE_BIAS_0 {-10, 6, 6, 4, 4, 3, -7, 6, 0, -4, -6, 4}

#define DENSE_BIAS_0_SHIFT (6)


#define DENSE_1_KERNEL_0 {63, -24, 29, 73, 53, 54, -79, -20, -64, -8, 20, 73, -55, -20, -19, -68, 53, 68, -45, -56, -71, -92, 51, 59, -36, 80, 4, 97, -12, 20, -11, -57, 7, 18, 23, -21, 1, -28, 54, 65, -59, -26, -2, -22, 41, 78, -71, 25, 61, -53, 31, 65, 47, 2, -7, 62, 12, 20, 8, 5, -9, -66, -43, 5, -67, -47, 85, 7, -69, -22, 67, 36, 19, 68, 31, -36, -73, -37, -67, -11, 55, -36, 59, -25, 72, 12, -25, 26, 74, -45, -64, 53, 14, -70, -22, -63}

#define DENSE_1_KERNEL_0_SHIFT (7)

#define DENSE_1_BIAS_0 {-64, 25, -40, -82, 71, 0, 70, -40}

#define DENSE_1_BIAS_0_SHIFT (11)


#define DENSE_2_KERNEL_0 {27, -3, -3, 60, -8, -26, -57, -53, -34, -52, -30, 30, -17, 6, 39, 25, -37, 7, 19, 78, 50, 34, -79, -53, 6, 36, 18, -35, -27, -30, 21, 3, -10, 27, 39, 84, -32, -5, -27, -21, -21, 48, 19, -42, 6, -16, 22, 20, 6, 2, 23, 53, -16, -21, -13, -32, 13, -4, 32, -3, 0, 21, -5, -19}

#define DENSE_2_KERNEL_0_SHIFT (6)

#define DENSE_2_BIAS_0 {0, -40, 67, 60, -78, -34, 0, 0}

#define DENSE_2_BIAS_0_SHIFT (9)


#define DENSE_3_KERNEL_0 {31, -52, 52, 51, -29, -66, -11, -2, 14, 70, 2, 25, 16, 63, -14, -41}

#define DENSE_3_KERNEL_0_SHIFT (6)

#define DENSE_3_BIAS_0 {117, -117}

#define DENSE_3_BIAS_0_SHIFT (10)



/* output enconding for each layer */
#define DENSE_INPUT_OUTPUT_SHIFT 0
#define DENSE_OUTPUT_SHIFT -3
#define ACTIVATION_OUTPUT_SHIFT 7
#define DENSE_1_OUTPUT_SHIFT 5
#define ACTIVATION_1_OUTPUT_SHIFT 5
#define DENSE_2_OUTPUT_SHIFT 4
#define ACTIVATION_2_OUTPUT_SHIFT 4
#define DENSE_3_OUTPUT_SHIFT 4
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
