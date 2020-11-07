package com.guyabihanna.neuralnet;

import java.util.ArrayList;

import Jama.Matrix;

public class Layer {
	Matrix inputs;
	Matrix weights;
	Matrix layerOutput;
	ArrayList<Neuron> layer;

	public Layer(Matrix inputs, Matrix initialWeights, ArrayList<Neuron> layer) {
		this.inputs = inputs;
		this.layer = layer;
		this.weights = initialWeights;
		layerOutput = new Matrix(layer.size(), 1);
	}
 
	public void updateNeuronsWeights() {
		for (int i = 0; i < layer.size(); i++) {
			layer.get(i).weights = weights;
		}
	}

	public void generateOutput() {
		for (int i = 0; i < layer.size(); i++) {
			double lOutput = layer.get(i).getNeuronOutput(i);
			layerOutput.set(i, 0, lOutput);
		}
	}

}
