#include "random.h"

using aurora::math::random;

random::random(unsigned int a_init) : dre(a_init) {

}

double random::next_double(double a_min, double a_max) {
	std::uniform_real_distribution<double> urd(a_min, a_max);
	return urd(dre);
}