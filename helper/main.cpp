#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("loss_distributor", helper::model) << std::endl;

	return 0;
}