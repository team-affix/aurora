#include "genome.h"

using aurora::optimization::genome;

genome::genome(tensor a_tens, function<genome(genome&)> a_mutate, function<double(genome&)> a_get_cost) : tensor(a_tens), mutate(a_mutate), get_cost(a_get_cost) {

}