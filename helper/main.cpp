#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("cnl_filter", helper::model) << std::endl;

	return 0;
}