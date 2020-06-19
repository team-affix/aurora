#pragma once
#include "main.h"
#include "general.h"


int main() {

	layerBpg l1 = layerBpg();
	layerBpg l2 = layerBpg();
	layerBpg l3 = layerBpg();


	rep([&l1]() {  
		seqBpg nlr = seqBpg();
		nlr.push_back(new biasBpg());
		nlr.push_back(new actBpg((actFunc*)new actFuncLR(0.05)));
		l1.push_back(nlr);
		}, 2);
	rep([&l2]() {
		seqBpg nlr = seqBpg();
		nlr.push_back(new biasBpg());
		nlr.push_back(new actBpg((actFunc*)new actFuncLR(0.05)));
		l2.push_back(nlr);
		}, 5);
	rep([&l3]() {
		seqBpg nlr = seqBpg();
		nlr.push_back(new biasBpg());
		nlr.push_back(new actBpg((actFunc*)new actFuncLR(0.05)));
		l3.push_back(nlr);
		}, 1);

	return 0;

}