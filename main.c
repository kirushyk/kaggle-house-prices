#include <stdio.h>
#include <le/le.h>

int main(int argc, char *argv[])
{
	LeTensor *features = le_matrix_new_zeros(1460, 80);
	LeTensor *prices = le_matrix_new_zeros(1460, 1);
	LeSequential *nn = le_sequential_new();
	le_sequential_add(nn, LE_LAYER(le_dense_layer_new("FC1", 80, 80)));
	le_sequential_add(nn, LE_LAYER(le_activation_layer_new("A1", LE_ACTIVATION_TANH)));
	le_sequential_add(nn, LE_LAYER(le_dense_layer_new("FC2", 80, 1)));
	LeBGD *optimizer = le_bgd_new(le_model_get_parameters(LE_MODEL(nn)), 0.3f);
	for (unsigned i = 0; i < 1000; i++)
	{
		LeList *grad = le_model_get_gradients(LE_MODEL(nn), features, prices);
		LE_OPTIMIZER(optimizer)->gradients = grad;
		le_optimizer_step(LE_OPTIMIZER(optimizer));
		le_list_foreach(grad, le_tensor_free);
	}
	le_bgd_free(optimizer);
	le_sequential_free(nn);
	return 0;
}

