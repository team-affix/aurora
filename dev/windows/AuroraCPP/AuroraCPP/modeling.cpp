#pragma once
#include "modeling.h"

#pragma region functions

#pragma region model
void modelFwd(sPtr<cType> x, sPtr<cType> y) {
	*y = *x;
}
void modelBwd(sPtr<cType> yGrad, sPtr<cType> xGrad) {
	*xGrad = *yGrad;
}
#pragma endregion
#pragma region bias
void biasFwd(sPtr<cType> x, sPtr<cType> y) {

}
void biasBwd(sPtr<cType> x, sPtr<cType> y) {

}
#pragma endregion

#pragma endregion


#pragma region definitions

#pragma region model
void model::fwd() {
	modelFwd(x, y);
}
void modelBpg::bwd() {
	modelBwd(yGrad, xGrad);
}
void model::modelWise(function<void(model*)> func) {
	func(this);
}
#pragma endregion
#pragma region bias
void bias::fwd() {
	biasFwd(x, y);
}
void biasBpg::fwd() {
	biasFwd(x, y);
}
void biasBpg::bwd() {
	biasBwd(yGrad, xGrad);
}
#pragma endregion

#pragma endregion