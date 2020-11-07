package com.guyabihanna.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Stack;

import com.jfoenix.controls.JFXTextField;

import Jama.Matrix;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.geometry.Insets;
import javafx.geometry.Point2D;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;
import javafx.util.Duration;

public class NeuralNetwork {
	ArrayList<Layer> neuralnet;

	public NeuralNetwork(ArrayList<Layer> neuralnet) {
		this.neuralnet = neuralnet;
	}

	public Matrix generateNetworkOutput(Matrix input) {
		for (int i = 0; i < neuralnet.size(); i++) { // Going through each layer
			Layer currentLayer = neuralnet.get(i);
			currentLayer.inputs = input;
			currentLayer.generateOutput();
			input = currentLayer.layerOutput;
		}
		return neuralnet.get(neuralnet.size() - 1).layerOutput;
	}

	ArrayList<Double> errorVals = new ArrayList<Double>();
	ArrayList<Integer> trainingItrts = new ArrayList<Integer>();
	int countIteration = 0;
	public void trainNetwork(Matrix input, Matrix desiredOut, double learningRate, double errorThreshold,
			String errorFunc, int trainingLimit) {
		// if (input.getRowDimension() != desiredOut.getRowDimension())
		// System.out.println("Training Input size different than Label Size");
		// int trainingLimit = 10;
		
		int i = 0;
		
		while (true) {
			Matrix neuralOutput = neuralnet.get(neuralnet.size() - 1).layerOutput;
			Matrix error = computeError(neuralOutput, desiredOut, errorThreshold, errorFunc);
			double errorVal = computeErrorVal(neuralOutput, desiredOut, errorThreshold, errorFunc);
			errorVals.add(errorVal);
			trainingItrts.add(++countIteration);
			boolean isTrained = (Math.pow((error.normF()), 2)) == 0;
			if (isTrained) {
				System.out.println("Trained");
				return;
			} else {
				computeNetWeights(input, desiredOut, neuralOutput, learningRate, errorThreshold, errorFunc);
				generateNetworkOutput(input);
				i++;
				if (i == trainingLimit) {
					System.out.println("Training Limit Reached.");
					break;
				}
			}
		}

	}

	private void computeNetWeights(Matrix input, Matrix desOut, Matrix realOutY, double learningRate,
			double errorThreshold, String errorFunc) {
		Matrix error_n = computeError(realOutY, desOut, errorThreshold, errorFunc);
		for (int i = neuralnet.size() - 1; i >= 0; i--) {
			Layer layer = neuralnet.get(i);
			// new weight = old weight + delta (set equal to old then add)
			Matrix currentWeight = layer.weights;
			System.out.println("v(t)");
			currentWeight.print(5, 1);
			Matrix deltaWeights = computeDeltaWeights(error_n, layer.inputs, learningRate);
			Matrix updatedWeights = currentWeight.plus(deltaWeights);
			System.out.println("v(t+1)");
			updatedWeights.print(5, 1);
			layer.weights = updatedWeights;
			layer.updateNeuronsWeights();
			error_n = currentWeight.times(error_n);
		}
	}

	private Matrix computeDeltaWeights(Matrix error, Matrix inputs, double learningRate) {
		Matrix delta = (error.times(inputs.transpose()).transpose()).times(learningRate);
		System.out.println("Error");
		error.print(5, 1);
		System.out.println("Inputs");
		inputs.print(5, 1);
		System.out.println("deltaV");
		delta.print(5, 1);
		return delta;
	}

	private double computeErrorVal(Matrix realOutputY, Matrix desOut, double errorThreshold, String errorFunc) {
		Matrix error = desOut.minus(realOutputY);
		double out = 0.0;
		switch (errorFunc) {
		case "MDE":
			double sum = 0;
			for (int i = 0; i < error.getRowDimension(); i++) {
				sum += error.get(i, 0);
			}
			out = (sum) / (error.getColumnDimension() * error.getRowDimension());
			break;
		case "MAE":
			out = (error.norm1()) / (error.getColumnDimension() * error.getRowDimension());
			break;

		case "MSE":
			out = (Math.pow(error.normF(), 2) / (error.getColumnDimension() * error.getRowDimension()));
			break;
		}
		return out;
	}
	
	private Matrix computeError(Matrix realOutputY, Matrix desOut, double errorThreshold, String errorFunc) {
		Matrix error = desOut.minus(realOutputY);
		switch (errorFunc) {
		case "MDE":
			double sum = 0;
			for (int i = 0; i < error.getRowDimension(); i++) {
				sum += error.get(i, 0);
			}
			double mde = (sum) / (error.getColumnDimension() * error.getRowDimension());
			if (mde <= errorThreshold)
				error = desOut.minus(desOut); // setting error to zero
			break;
		case "MAE":
			double mae = (error.norm1()) / (error.getColumnDimension() * error.getRowDimension());
			if (mae <= errorThreshold)
				error = desOut.minus(desOut);
			break;

		case "MSE":
			double mse = (Math.pow(error.normF(), 2) / (error.getColumnDimension() * error.getRowDimension()));
			if (mse <= errorThreshold)
				error = desOut.minus(desOut);
			break;
		}
		return error;
	}

	public static ArrayList<ArrayList<Double>> prettyWeights(ArrayList<Layer> layers) {

		ArrayList<ArrayList<Double>> weights = new ArrayList<>();
		for (int i = 0; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			Matrix layerWeights = layer.weights;
			double[] neuronWeights = layerWeights.getColumnPackedCopy();
			Double[] neuronWts = new Double[neuronWeights.length];
			for (int k = 0; k < neuronWeights.length; k++)
				neuronWts[k] = Double.valueOf(neuronWeights[k]);
			weights.add(new ArrayList<>(Arrays.asList(neuronWts)));
		}
		return weights;
	}

	public static Point2D getDirection(Circle c1, Circle c2) {
		return new Point2D(c2.getCenterX() - c1.getCenterX(), c2.getCenterY() - c1.getCenterY()).normalize();
	}

	public void drawNeuron(ArrayList<Double> inputs, Pane pane, Matrix inputMx) {

		ArrayList<ArrayList<Double>> networkWeights = prettyWeights(neuralnet);
		ArrayList<Label> labelArr = new ArrayList<Label>();
		ArrayList<Circle> inputsC = new ArrayList<Circle>();

		// networkWeights.add(0, inputs);

		for (int i = 0; i < inputs.size(); i++) {
			double centerX = 30;
			double centerYMin = 80;
			double centerYSep = 50;
			double radius = 15;
			Color fill = Color.BLACK;
			Circle c = new Circle(centerX, centerYMin + (centerYSep * i), radius, fill);
			c.setStroke(Color.BLACK);
			c.setStrokeWidth(3);
			Label lbl = new Label();
			lbl.setText(String.format("%.2f", inputs.get(i)));
			lbl.setTextFill(Color.WHITE);
			lbl.setLayoutX(c.getCenterX() - radius / 2);
			lbl.setLayoutY(c.getCenterY() - radius / 2);
			lbl.setTextAlignment(TextAlignment.LEFT);
			lbl.setBackground(new Background(new BackgroundFill(Color.BLACK, CornerRadii.EMPTY, Insets.EMPTY)));
			inputsC.add(c);
			pane.getChildren().add(c);
			pane.getChildren().add(lbl);
			labelArr.add(lbl);
			int currentInputNb = i;
			c.setOnMouseClicked(e -> {
				Stage s = new Stage();

				GridPane p = new GridPane();
				p.setHgap(8);
				p.setVgap(10);
				JFXTextField txt = new JFXTextField("Input Value: ");
				txt.setEditable(false);
				p.add(txt, 0, 0);
				JFXTextField txt2 = new JFXTextField();
				txt2.setText("" + inputs.get(currentInputNb));
				txt.setEditable(false);
				p.add(txt2, 1, 0);
				Button btnSave = new Button("Save");
				btnSave.setMaxWidth(Double.MAX_VALUE);
				p.add(btnSave, 1, 1);
				// i is which layer we are on
				// j is which neuron we are on
				// n is the neuron we are on

				btnSave.setOnMouseClicked(event -> {
					double newIn = Double.parseDouble(txt2.getText());
					double[][] arr = inputMx.getArray();
					arr[currentInputNb][0] = newIn;
					Matrix m = new Matrix(arr);
					ArrayList<Double> inn = toArrayList(arr);

					generateNetworkOutput(m);
					drawNeuron(inn, pane, m);
					s.close();
					
				});

				Scene scene = new Scene(p, p.getMaxWidth(), p.getMaxHeight());
				s.setScene(scene);
				s.setResizable(true);
				s.setTitle("Input");
				s.show();
			});

		}

		// ArrayList<ArrayList<Integer>> layers = new ArrayList<ArrayList<Integer>>();

		// 3 layers with 5 neurons maximum in each layer
		for (int i = 0; i < 5; i++) {
			ArrayList<Integer> arr = new ArrayList<Integer>();
			Random rnd = new Random();
			int maxNbOfNeurons = rnd.nextInt(10) + 1;
			for (int j = 0; j < maxNbOfNeurons; j++) {
				arr.add(j);
			}
			// layers.add(arr);
		}

		ArrayList<ArrayList<Circle>> allCircles = new ArrayList<ArrayList<Circle>>();
		allCircles.add(inputsC); // Add input Circles

		for (int i = 0; i < neuralnet.size(); i++) {
			Layer layer = neuralnet.get(i);
			Random rnd = new Random();
			int red = rnd.nextInt(256);
			int green = rnd.nextInt(256);
			int blue = rnd.nextInt(256);
			Color fill = Color.rgb(red, green, blue); // each layer has a color
			ArrayList<Circle> layerCircles = new ArrayList<Circle>();
			for (int j = 0; j < layer.layer.size(); j++) {
				double centerX = 190;
				double centerXSep = 160;
				double centerYMin = 100;
				double centerYSep = 100;
				double radius = 35;
				Circle n = new Circle(centerX + (i * centerXSep), centerYMin + (j * centerYSep), radius, fill);
				n.setStroke(Color.BLACK);
				n.setStrokeWidth(3);
				layerCircles.add(n);
				pane.getChildren().add(n);
			}
			allCircles.add(layerCircles);
		}

		Timeline timeline = new Timeline();
		int temp = 0;
		int temp2 = 1;
		for (int i = 0; i < allCircles.size(); i++) {
			ArrayList<Circle> layerCircles = allCircles.get(i);
			Layer layers = null;
			if (temp >= 1) {
				layers = neuralnet.get(i - 1);
			} else {
				layers = neuralnet.get(0);
			}
			temp++;
			Random rnd = new Random();
			int red = rnd.nextInt(256);
			int green = rnd.nextInt(256);
			int blue = rnd.nextInt(256);
			Color fill = Color.rgb(red, green, blue);
			int count = 0;
			for (int j = 0; j < layerCircles.size(); j++) {
				// If we are NOT on the final layer we can add lines
				if (i < (allCircles.size() - 1)) {
					// This circle is the input
					// System.out.println(layerCircles.size());
					// System.out.println(layers.layer.size());
					Neuron n = null;
					try {
						if (count >= 1) {
							n = layers.layer.get(j - 1);
						} else {
							n = layers.layer.get(0);
						}
					} catch (Exception e) {

					}
					Circle c = layerCircles.get(j);
					count++;
					ArrayList<Circle> nextLayerCircles = allCircles.get(i + 1);
					Layer nextLayer = null;
					if (temp2 >= 1) {
						nextLayer = neuralnet.get(i);
					} else {
						nextLayer = neuralnet.get(1);
					}
					temp2++;
					for (int w = 0; w < nextLayerCircles.size(); w++) {

						Line l = new Line();
						Circle cc = nextLayerCircles.get(w);
						Neuron nn = nextLayer.layer.get(w);
						if (count > 1) {

							String cssLayout = "-fx-background-color: white;\n" + "-fx-background-radius: 10;\n"
									+ "-fx-border-radius: 10;\n" + "-fx-border-color: black;\n"
									+ "-fx-border-width: 2;\n";
							VBox box = new VBox();
							box.setLayoutX(cc.getCenterX());
							box.setLayoutY(cc.getCenterY() - cc.getRadius());
							box.setBackground(
									new Background(new BackgroundFill(Color.WHITE, CornerRadii.EMPTY, Insets.EMPTY)));
							box.setStyle(cssLayout);
							Label lbl2 = new Label();
							lbl2.setTextFill(Color.BLACK);
							lbl2.setText("•Activation: " + nn.actFunc);
							box.getChildren().add(lbl2);
							Label lbl3 = new Label();
							lbl3.setTextFill(Color.BLACK);
							lbl3.setText("•Bias: " + nn.bias);
							box.getChildren().add(lbl3);
							Label lbl = new Label();
							// lbl.setLayoutX(cc.getCenterX() + cc.getRadius());
							// lbl.setLayoutY(cc.getCenterY() - 1.2 * cc.getRadius());
							lbl.setText("•Output: " + nn.output);
							lbl.setTextFill(Color.BLACK);
							lbl.setTextAlignment(TextAlignment.CENTER);

							box.getChildren().add(lbl);
							Timeline t = new Timeline();
							t.getKeyFrames()
									.add(new KeyFrame(Duration.ZERO, new KeyValue(lbl.visibleProperty(), false)));
							t.play();
							// pane.getChildren().add(lbl);

							// labelArr.add(lbl);
							// pane.getChildren().add(box);
							// cc.setOnMouseEntered(e -> pane.getChildren().add(lbl));
							// cc.setOnMouseEntered(e -> labelArr.add(lbl));
							// cc.setOnMouseEntered(e -> pane.getChildren().add(box));

							cc.setOnMouseEntered(e -> pane.getChildren().add(box));
							cc.setOnMouseExited(e -> pane.getChildren().remove(box));

							// cc.setOnMouseEntered(e -> removeLabel(addLabel(cc,pane,labelArr, nn)));
						}
						timeline.getKeyFrames().add(new KeyFrame(Duration.seconds((i + 2)),
								new KeyValue(l.startXProperty(), cc.getCenterX())));
						timeline.getKeyFrames().add(new KeyFrame(Duration.seconds((i + 2)),
								new KeyValue(l.startYProperty(), cc.getCenterY())));
						timeline.getKeyFrames().add(new KeyFrame(Duration.seconds((i + 2)),
								new KeyValue(l.endXProperty(), c.getCenterX())));
						timeline.getKeyFrames().add(new KeyFrame(Duration.seconds((i + 2)),
								new KeyValue(l.endYProperty(), c.getCenterY())));

						timeline.getKeyFrames()
								.add(new KeyFrame(Duration.ZERO, new KeyValue(l.startXProperty(), c.getCenterX())));
						timeline.getKeyFrames()
								.add(new KeyFrame(Duration.ZERO, new KeyValue(l.startYProperty(), c.getCenterY())));
						timeline.getKeyFrames()
								.add(new KeyFrame(Duration.ZERO, new KeyValue(l.endXProperty(), c.getCenterX())));
						timeline.getKeyFrames()
								.add(new KeyFrame(Duration.ZERO, new KeyValue(l.endYProperty(), c.getCenterY())));
						l.setStrokeWidth(3);
						l.setStartX(c.getCenterX());
						l.setEndX(nextLayerCircles.get(w).getCenterX());
						l.setStartY(c.getCenterY());
						l.setEndY(nextLayerCircles.get(w).getCenterY());
						pane.getChildren().add(l);
						timeline.getKeyFrames()
								.add(new KeyFrame(Duration.seconds(i + 1), new KeyValue(c.fillProperty(), fill)));
						timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(i + 1),
								new KeyValue(nextLayerCircles.get(w).fillProperty(), fill)));
						timeline.getKeyFrames()
								.add(new KeyFrame(Duration.ZERO, new KeyValue(c.fillProperty(), Color.TRANSPARENT)));
						timeline.getKeyFrames().add(new KeyFrame(Duration.ZERO,
								new KeyValue(nextLayerCircles.get(w).fillProperty(), Color.TRANSPARENT)));
						c.toFront();
						nextLayerCircles.get(w).toFront();

					}
				}
			}
		}
		timeline.play();

		for (int i = 0; i < neuralnet.size(); i++) {
			Layer l = neuralnet.get(i);
			ArrayList<Circle> layerCircles = allCircles.get(i + 1);
			ArrayList<Neuron> neuronsInLayer = l.layer;
			for (int j = 0; j < neuronsInLayer.size(); j++) {
				final int atNeuronNb = j;
				Circle cc = layerCircles.get(j);
				Neuron n = neuronsInLayer.get(j);

				double[][] arr = n.weights.getArray();
				ArrayList<Double> neuronWeights = getColumn(arr, j);
				for (int a = 0; a < neuronWeights.size(); a++) {
					Label lbl = new Label();
					
					lbl.setText(String.format("%.2f", neuronWeights.get(a)));
					lbl.setTextFill(Color.WHITE);
					lbl.setLayoutX(cc.getCenterX() - cc.getRadius() * 1.2);
					lbl.setLayoutY(cc.getCenterY() - a * 20);
					lbl.setTextAlignment(TextAlignment.CENTER);
					lbl.setBackground(new Background(new BackgroundFill(Color.BLACK, CornerRadii.EMPTY, Insets.EMPTY)));
					pane.getChildren().add(lbl);
					labelArr.add(lbl);
				}

				cc.setOnMouseClicked(e -> {
					Stage s = new Stage();

					GridPane p = new GridPane();
					p.setHgap(8);
					ArrayList<JFXTextField> updatedWeightsTxt = new ArrayList<JFXTextField>();
					for (int q = 0; q < neuronWeights.size(); q++) {
						JFXTextField t = new JFXTextField();
						updatedWeightsTxt.add(t);
						t.setText("" + flip(neuronWeights).get(q));
						JFXTextField txt = new JFXTextField("Weight# " + (q + 1));
						txt.setEditable(false);
						p.add(txt, 0, q);
						p.add(t, 1, q);
					}

					p.setVgap(10);
					JFXTextField txt = new JFXTextField("Activation Function");
					txt.setEditable(false);
					p.add(txt, 0, neuronWeights.size() + 2);
					JFXTextField actFuncTxt = new JFXTextField(n.actFunc);
					p.add(actFuncTxt, 1, neuronWeights.size() + 2);
					txt = new JFXTextField("Bias");
					txt.setEditable(false);
					p.add(txt, 0, neuronWeights.size() + 4);
					JFXTextField biasTxt = new JFXTextField("" + n.bias);
					p.add(biasTxt, 1, neuronWeights.size() + 4);
					Button btnSave = new Button("Save");
					btnSave.setMaxWidth(Double.MAX_VALUE);
					p.add(btnSave, 1, neuronWeights.size() + 5);
					// i is which layer we are on
					// j is which neuron we are on
					// n is the neuron we are on

					btnSave.setOnMouseClicked(event -> {
						double newBias = Double.parseDouble(biasTxt.getText());
						String newActFnc = actFuncTxt.getText();
						n.bias = newBias;
						n.actFunc = newActFnc;
						ArrayList<Double> updatedWeights = new ArrayList<Double>();
						// System.out.println("New bias: " + newBias);
						// System.out.println("New Act Func: " + newActFnc);

						for (int r = 0; r < updatedWeightsTxt.size(); r++) {
							updatedWeights.add(Double.parseDouble(updatedWeightsTxt.get(r).getText()));
							// System.out.println("" + updatedWeights.get(r));
						}

						for (int a = 0; a < arr.length; a++) {
							// System.out.println("Replacing: ");
							// System.out.print(arr[a][atNeuronNb] + " " + "By");
							// System.out.println(""+updatedWeights.get(a));
							arr[a][atNeuronNb] = updatedWeights.get(a);
						}
						n.weights = new Matrix(arr);
						generateNetworkOutput(inputMx);
						drawNeuron(inputs, pane, inputMx);
						s.close();
					});

					Scene scene = new Scene(p, p.getMaxWidth(), p.getMaxHeight());
					s.setScene(scene);
					s.setResizable(true);
					s.setTitle("Neuron");
					s.show();
				});

			}

		}

		ArrayList<Circle> layerCircles = allCircles.get(allCircles.size() - 1);
		ArrayList<Double> lastLayerOutput = new ArrayList<Double>();
		for (int i = 0; i < neuralnet.get(neuralnet.size() - 1).layer.size(); i++) {
			Neuron n = neuralnet.get(neuralnet.size() - 1).layer.get(i);
			lastLayerOutput.add(n.output);
		}

		for (int i = 0; i < layerCircles.size(); i++) {
			Circle c = layerCircles.get(i);
			double centerX = layerCircles.get(0).getCenterX();
			double centerXSep = 160;
			double centerYMin = layerCircles.get(i).getCenterY();
			;
			double centerYSep = 0;
			double radius = 10;
			Circle n = new Circle(centerX + centerXSep, centerYMin + (i * centerYSep), radius, Color.BLACK);
			n.setStroke(Color.BLACK);
			n.setStrokeWidth(3);
			Line l = new Line();
			l.setStrokeWidth(3);
			l.setStartX(c.getCenterX());
			l.setStartY(c.getCenterY());
			l.setEndX(n.getCenterX());
			l.setEndY(n.getCenterY());
			Label lbl = new Label();

			lbl.setText(String.format("%.2f", lastLayerOutput.get(i)));
			lbl.setTextFill(Color.WHITE);
			lbl.setLayoutX(l.getEndX() - n.getRadius() * 3);
			lbl.setLayoutY(l.getEndY() - 2 * n.getRadius());
			lbl.setTextAlignment(TextAlignment.CENTER);
			lbl.setBackground(new Background(new BackgroundFill(Color.BLACK, CornerRadii.EMPTY, Insets.EMPTY)));
			pane.getChildren().add(lbl);
			pane.getChildren().add(l);
			pane.getChildren().add(n);

			c.toFront();

		}

		for (int i = 0; i < labelArr.size(); i++) {
			labelArr.get(i).toFront();
		}

	}

	public static ArrayList<Double> getColumn(double[][] arr, int index) {
		ArrayList<Double> out = new ArrayList<Double>();
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[0].length; j++) {

			}
			out.add(arr[i][index]);
		}
		Stack<Double> s = new Stack<Double>();
		for (int i = 0; i < out.size(); i++) {
			s.push(out.get(i));
		}
		out = new ArrayList<Double>();
		while (!s.isEmpty()) {
			out.add(s.pop());
		}

		return out;
	}

	public static ArrayList<Double> flip(ArrayList<Double> arr) {
		ArrayList<Double> out = new ArrayList<Double>();

		Stack<Double> s = new Stack<Double>();
		for (int i = 0; i < arr.size(); i++) {
			s.push(arr.get(i));
		}
		out = new ArrayList<Double>();
		while (!s.isEmpty()) {
			out.add(s.pop());
		}

		return out;
	}

	public static ArrayList<Double> toArrayList(double[][] arr) {
		ArrayList<Double> out = new ArrayList<Double>();
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[0].length; j++) {
				out.add(arr[i][j]);
			}
		}
		return out;
	}

}
