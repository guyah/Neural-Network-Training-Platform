package com.guyabihanna.neuralnet;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import Jama.Matrix;

public class Neuron {

	Matrix inputs;
	Matrix weights;
	double output;
	double bias;
	String actFunc;

	public Neuron(Matrix inputs, Matrix weights, double bias, String actFunc) {
		this.inputs = inputs;
		this.weights = weights;
		this.bias = bias;
		this.actFunc = actFunc;
		this.output = 0;
		if (inputs.getRowDimension() != weights.getRowDimension()) {
			System.out.println("Error: Input Matrix size different that Weight Matrix Size");
		}

	}

	public double calculateNeuronOutput(int neuron) {
		double output;
		int[] colInd = { neuron };
		Matrix neuronWeight = weights.getMatrix(0, weights.getRowDimension() - 1, colInd);
		output = neuronWeight.transpose().times(inputs).det();
		output -= bias;
		this.output = output;
		return output;
	}

	public double getNeuronOutput(int neuron) {
		double actFuncParameter = calculateNeuronOutput(neuron);
		switch (actFunc) {
		case "step":
			output = unitFunction(actFuncParameter);
			break;
		case "linear":
			output = linear(actFuncParameter);
			break;
		case "sigmoid":
			output = sigmoid(actFuncParameter);
			break;
		default:
			if (actFunc.matches("gaussian.*")) {
				Pattern pattern = Pattern.compile("\\d+.\\d+");
				Matcher matcher = pattern.matcher(actFunc);
				double[] xsigma = new double[2];
				int i = 0;
				while (matcher.find()) {
					xsigma[i] = Double.parseDouble(matcher.group());
					i++;
				}
				output = gaussian(xsigma[0], xsigma[1]);
			} else if (actFunc.matches("boundedLinear.*")) {
				Pattern pattern = Pattern.compile("\\d+.\\d+");
				Matcher matcher = pattern.matcher(actFunc);
				double[] minmax = new double[2];
				int i = 0;
				while (matcher.find()) {
					minmax[i] = Double.parseDouble(matcher.group());
					i++;
				}
				output = boundedLinear(minmax[0], minmax[1]);
			}

		}
		return output;

	}

	private double unitFunction(double val) {
		if (val <= 0) {
			val = 0.0;
			return 0.0;
		}
		val = 1.0;
		return 1.0;
	}

	private double boundedLinear(double min, double max) {
		if (output < min) {
			return 0.0;
		} else if (output >= min && output <= max) {
			double out = ((output - min) / (max - min));
			return out;
		} else {
			this.output = 1.0;
			return 1.0;
		}
	}

	private double linear(double val) {
		return val;
	}

	private double sigmoid(double val) {

		double out = 1 / (1 + Math.exp(-val));
		return out;

	}

	private double gaussian(double x, double sigma) {
		double out = Math.exp(((output - x) / sigma));
		return out;
	}

}
