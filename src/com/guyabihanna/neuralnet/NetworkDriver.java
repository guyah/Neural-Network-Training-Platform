package com.guyabihanna.neuralnet;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.jfoenix.controls.JFXButton;
import com.jfoenix.controls.JFXComboBox;
import com.jfoenix.controls.JFXTextField;

import Jama.Matrix;
import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Rectangle2D;
import javafx.geometry.VPos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Screen;
import javafx.stage.Stage;

public class NetworkDriver extends Application {

	public static void main(String[] args) {

		launch(args);

	}

	int nbOfLayers = 0;
	ArrayList<JFXTextField> nbrOfNrnsPerLyr = new ArrayList<>();
	NeuralNetwork neural = null;
	double[][] inputsArr2 = new double[1][1];
	double[][] desiredArr2 = new double[1][1];
	String cssButton = "-fx-background-color: white;\n" + "-fx-background-radius: 10;\n" + "-fx-border-radius: 10;\n"
			+ "-fx-border-color: black;\n" + "-fx-border-width: 2;\n";

	@Override
	public void start(Stage primaryStage) throws Exception {
		Stage stage = new Stage();
		GridPane gp = new GridPane(); // For whole application
		GridPane inPane = new GridPane(); // for input Pane
		inPane.setPrefWidth(300);
		Pane pane = new Pane(); // For neural Network
		ScrollPane ntwrkScrl = new ScrollPane();
		// gp.setBackground(new Background(new BackgroundFill(Color.WHITE,
		// CornerRadii.EMPTY, Insets.EMPTY)));
		// inPane.setBackground(new Background(new BackgroundFill(Color.SILVER,
		// CornerRadii.EMPTY, Insets.EMPTY)));
		// pane.setBackground(new Background(new BackgroundFill(Color.WHITE,
		// CornerRadii.EMPTY, Insets.EMPTY)));

		// Number of inputs
		// Number of layers
		// Number of neurons for each layer
		ScrollPane scrlPane = new ScrollPane();
		TitledPane ttlPane = new TitledPane();
		GridPane initGrid = new GridPane();
		initGrid.setVgap(8);
		initGrid.setHgap(8);
		initGrid.setPadding(new Insets(12, 0, 12, 0));

		ttlPane.setText("Initialization");
		ttlPane.setExpanded(false);
		ttlPane.setPrefHeight(400);
		scrlPane.setPrefWidth(inPane.getPrefWidth());
		VBox box = new VBox(5);
		HBox inputBox = new HBox(5);

		Label inputlb = new Label("Number of inputs: ");
		JFXTextField inputTF = new JFXTextField();
		inputTF.setStyle("-fx-background-color: white;");
		inputTF.setPrefWidth(40);
		initGrid.add(inputlb, 0, 1);
		initGrid.add(inputTF, 1, 1);
		// inputBox.getChildren().addAll(inputlb, inputTF);
		// box.getChildren().add(inputBox);

		Label nbLayers = new Label("Number of Layers: ");
		JFXTextField nbLayersTf = new JFXTextField();
		nbLayersTf.setStyle("-fx-background-color: white;");
		nbLayersTf.setPrefWidth(40);
		initGrid.add(nbLayers, 0, 2);
		initGrid.add(nbLayersTf, 1, 2);
		HBox lBox = new HBox(5);

		Label lbl_weightsGen = new Label("Generation of weights: ");
		JFXTextField tf_weightsGen = new JFXTextField();
		ObservableList<Object> ol_weightsGen = FXCollections.observableArrayList("Default (1.0)", "Random");
		JFXComboBox cb_weightsGen = new JFXComboBox(ol_weightsGen);
		initGrid.add(lbl_weightsGen, 0, 3);
		initGrid.add(cb_weightsGen, 1, 3);

		Label lbl_actFunc = new Label("Initial Act Func: ");
		JFXTextField tf_actFunc = new JFXTextField();
		ObservableList<Object> ol_actFunc = FXCollections.observableArrayList("step", "sigmoid", "linear",
				"Default bounded linear", "Default Gaussian");
		JFXComboBox cb_actFunc = new JFXComboBox(ol_actFunc);
		initGrid.add(lbl_actFunc, 0, 4);
		initGrid.add(cb_actFunc, 1, 4);
		// HBox lBox = new HBox(5);
		// lBox.getChildren().addAll(nbLayers, nbLayersTf);
		// box.getChildren().add(lBox);

		nbLayersTf.setOnAction(e1 -> {
			nbOfLayers = Integer.parseInt(nbLayersTf.getText());

			for (int i = 0; i < nbOfLayers; i++) {
				Label layerLabel = new Label("Layer " + (i + 1) + ": ");
				box.getChildren().addAll(layerLabel);
				HBox Nbox = new HBox(1.5);
				Label Nlabel = new Label("Layer " + (i + 1) + "\n" + "Number of Neurons: ");
				JFXTextField NTf = new JFXTextField();
				NTf.setStyle("-fx-background-color: white;");
				NTf.setPrefWidth(40);
				// initGrid.add(layerLabel, 0, 2+i);
				initGrid.add(Nlabel, 0, 4 + i + 2);
				initGrid.add(NTf, 1, 4 + i + 2);
				nbrOfNrnsPerLyr.add(NTf);
				// Nbox.getChildren().addAll(Nlabel, NTf);
				// box.getChildren().addAll(Nbox);
			}

			JFXButton generateNetworkBtn = new JFXButton("Generate Network");
			generateNetworkBtn.setStyle(cssButton);
			initGrid.add(generateNetworkBtn, 0, 5 + nbOfLayers + 2);

			// box.getChildren().add(generateNetworkBtn);
			generateNetworkBtn.setOnMouseClicked(event -> {
				// input baddo yeh arraylist w bdde yeh 2d array w l data stored in a column
				// badde pane ta ersom 3laya l nn (3ande yeha)
				double initInput = 1.0;
				double initBias = 0.0;
				double initWeight = 1.0;
				String initAct = "step";

				// Generating inputs
				int nbOfInputs = Integer.parseInt(inputTF.getText());
				inputsArr2 = new double[Integer.parseInt(inputTF.getText())][1];

				double[][] inputsArr = new double[nbOfInputs][1];
				for (int i = 0; i < nbOfInputs; i++) {
					inputsArr[i][0] = initInput;
				}
				// Generating weights
				int nbOfLayers = Integer.parseInt(nbLayersTf.getText());
				ArrayList<double[][]> weights = new ArrayList<double[][]>();
				Matrix m1 = null;

				// System.out.println("Size: " + nbrOfNrnsPerLyr.size());
				if (cb_weightsGen.getValue().equals("Default (1.0)")) {
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						double[][] weightsArr = new double[nbOfInputs][nbOfNeuronsInLayer];
						for (int j = 0; j < weightsArr.length; j++) {
							for (int k = 0; k < weightsArr[0].length; k++) {
								weightsArr[j][k] = initWeight;
							}
						}
						m1 = new Matrix(weightsArr);
						m1.print(5, 1);
						nbOfInputs = nbOfNeuronsInLayer;
						weights.add(weightsArr);
					}
				} else {
					Random rnd = new Random();
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						double[][] weightsArr = new double[nbOfInputs][nbOfNeuronsInLayer];
						for (int j = 0; j < weightsArr.length; j++) {
							for (int k = 0; k < weightsArr[0].length; k++) {
								initWeight = rnd.nextInt(21) - 10;
								weightsArr[j][k] = initWeight;
							}
						}
						m1 = new Matrix(weightsArr);
						m1.print(5, 1);
						nbOfInputs = nbOfNeuronsInLayer;
						weights.add(weightsArr);
					}
				}
				int nbOfOutputs = Integer.parseInt(nbrOfNrnsPerLyr.get(nbrOfNrnsPerLyr.size() - 1).getText());
				desiredArr2 = new double[nbOfOutputs][1];
				// Generating bias
				ArrayList<ArrayList<Double>> biases = new ArrayList<ArrayList<Double>>();
				for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
					int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
					ArrayList<Double> bias = new ArrayList<Double>();
					for (int j = 0; j < nbOfNeuronsInLayer; j++) {
						bias.add(initBias);
					}
					biases.add(bias);
				}
				// Generating activation Functions
				ArrayList<ArrayList<String>> actFuncs = new ArrayList<ArrayList<String>>();
				if (cb_actFunc.getValue().equals("step")) {
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						ArrayList<String> actFunc = new ArrayList<String>();
						for (int j = 0; j < nbOfNeuronsInLayer; j++) {
							actFunc.add("step");
						}
						actFuncs.add(actFunc);

					}
				} 
				if (cb_actFunc.getValue().equals("sigmoid")) {
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						ArrayList<String> actFunc = new ArrayList<String>();
						for (int j = 0; j < nbOfNeuronsInLayer; j++) {
							actFunc.add("sigmoid");
						}
						actFuncs.add(actFunc);

					}
				} 
				if (cb_actFunc.getValue().equals("linear")) {
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						ArrayList<String> actFunc = new ArrayList<String>();
						for (int j = 0; j < nbOfNeuronsInLayer; j++) {
							actFunc.add("linear");
						}
						actFuncs.add(actFunc);

					}
				} 
				if (cb_actFunc.getValue().equals("Default bounded linear")) {
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						ArrayList<String> actFunc = new ArrayList<String>();
						for (int j = 0; j < nbOfNeuronsInLayer; j++) {
							actFunc.add("boundedLinear 0.0,2.0");
						}
						actFuncs.add(actFunc);

					}
				} 
				if (cb_actFunc.getValue().equals("Default Gaussian")) {
					for (int i = 0; i < nbrOfNrnsPerLyr.size(); i++) {
						int nbOfNeuronsInLayer = Integer.parseInt(nbrOfNrnsPerLyr.get(i).getText());
						ArrayList<String> actFunc = new ArrayList<String>();
						for (int j = 0; j < nbOfNeuronsInLayer; j++) {
							actFunc.add("gaussian 2.0,5.0");
						}
						actFuncs.add(actFunc);

					}
				} 
				
				
				
				ArrayList<Layer> layers = initializeNetwork(inputsArr2, weights, biases, actFuncs);
				NeuralNetwork nn = new NeuralNetwork(layers);
				neural = nn;
				ArrayList<Double> arrIn = toArrayList(inputsArr2);

				// nn.generateNetworkOutput(new Matrix(inputsArr));

				nn.drawNeuron(arrIn, pane, new Matrix(inputsArr2));
				gp.autosize();
				stage.sizeToScene();
			});

			// Saving the weight

			JFXButton btn_saveWeights = new JFXButton("Save Network");
			btn_saveWeights.setStyle(cssButton);
			JFXTextField tf_saveWeights = new JFXTextField();
			tf_saveWeights.setPromptText("File name");
			initGrid.add(tf_saveWeights, 1, 3 + nbOfLayers + 3);
			initGrid.add(btn_saveWeights, 0, 3 + nbOfLayers + 3);

			btn_saveWeights.setOnMouseClicked(event -> {
				try {
					if (tf_saveWeights.getText().length() != 0) {
						DirectoryChooser chooser = new DirectoryChooser();
						chooser.setTitle("Save Weights");
						File selectedDirectory = chooser.showDialog(primaryStage);
						ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(
								new File(selectedDirectory.toString() + "/" + tf_saveWeights.getText() + ".ann")));

						Integer layerSize = neural.neuralnet.size();
						os.writeObject(layerSize);
						for (int i = 0; i < neural.neuralnet.size(); i++) {
							Layer lyr = neural.neuralnet.get(i);
							os.writeObject(lyr.layer.size());
						}

						for (int i = 0; i < neural.neuralnet.size(); i++) {
							Layer lyr = neural.neuralnet.get(i);
							Matrix w1 = lyr.weights;
							os.writeObject(w1);
							for (int j = 0; j < lyr.layer.size(); j++) {
								System.out.println("ok");
								Neuron nrn = lyr.layer.get(j);
								// Activation Function
								String acFnc = nrn.actFunc;
								os.writeObject(acFnc);
								// Bias
								Double biasS = nrn.bias;
								os.writeObject(biasS);
							}
						}

						Integer nbrOfInputs = neural.neuralnet.get(0).inputs.getRowDimension();
						os.writeObject(nbrOfInputs);
					} else {
						Alert alert = new Alert(AlertType.ERROR, "Enter a file name", ButtonType.OK);
						alert.showAndWait();
					}
				} catch (Exception exc) {
					exc.printStackTrace();
				}
			});

		});

		JFXButton btn_loadWeights = new JFXButton("Load Network");
		btn_loadWeights.setStyle(cssButton);
		initGrid.add(btn_loadWeights, 0, 0);

		btn_loadWeights.setOnMouseClicked(event -> {
			try {
				FileChooser chooser = new FileChooser();
				chooser.setTitle("Weights File");
				File weightsFile = chooser.showOpenDialog(primaryStage);
				ObjectInputStream is = new ObjectInputStream(new FileInputStream(weightsFile));

				Integer layerSize = (Integer) is.readObject();
				ArrayList<Integer> neuronsPerLayer = new ArrayList<Integer>();
				for (int i = 0; i < layerSize; i++) {
					neuronsPerLayer.add((Integer) is.readObject());
				}

				ArrayList<double[][]> loadWeights = new ArrayList<>();
				ArrayList<ArrayList<String>> loadActFnc = new ArrayList<ArrayList<String>>();
				ArrayList<ArrayList<Double>> loadBiases = new ArrayList<ArrayList<Double>>();

				for (int i = 0; i < layerSize; i++) {
					Matrix m1 = (Matrix) is.readObject();
					m1.print(5, 1);
					System.out.println("Weight");
					loadWeights.add(m1.getArray());
					ArrayList<String> loadLayerAct = new ArrayList<String>();
					ArrayList<Double> loadBias = new ArrayList<Double>();

					for (int j = 0; j < neuronsPerLayer.get(i); j++) {
						String actFnc = (String) is.readObject();
						System.out.println("Act Func: " + actFnc);
						loadLayerAct.add(actFnc);
						Double bias = (Double) is.readObject();
						System.out.println("Bias: " + bias);
						loadBias.add(bias);

					}

					loadActFnc.add(loadLayerAct);
					loadBiases.add(loadBias);
				}
				Integer nbOfInputs = (Integer) is.readObject();
				inputsArr2 = new double[nbOfInputs][1];

				ArrayList<Layer> layers = initializeNetwork(inputsArr2, loadWeights, loadBiases, loadActFnc);
				NeuralNetwork nn = new NeuralNetwork(layers);
				neural = nn;
				neural.drawNeuron(toArrayList(inputsArr2), pane, new Matrix(inputsArr2));
				int nbOfOutputs = nn.neuralnet.get(nn.neuralnet.size() - 1).layerOutput.getRowDimension();
				desiredArr2 = new double[nbOfOutputs][1];
				gp.autosize();
				stage.sizeToScene();
			} catch (Exception exc) {
				exc.printStackTrace();
			}
		});

		TitledPane ttlPane2 = new TitledPane();
		ttlPane2.setText("Training");
		ttlPane2.setExpanded(false);
		ttlPane2.setPrefHeight(400);
		ScrollPane scrlPane2 = new ScrollPane();

		GridPane trngGrid = new GridPane();
		trngGrid.setVgap(8);
		trngGrid.setHgap(8);
		trngGrid.setPadding(new Insets(8, 0, 8, 0));

		VBox trVBox = new VBox(5);
		HBox trHBox = new HBox(5);
		Label lbl_learnRate = new Label("Learning rate");
		JFXTextField tf_learnRate = new JFXTextField();
		Label lbl_errorThresh = new Label("Error Threshold");
		JFXTextField tf_errorThresh = new JFXTextField();
		Label lbl_errorFunc = new Label("Error Function");
		ObservableList<String> ol_errorFunc = FXCollections.observableArrayList("MDE", "MAE", "MSE");
		JFXComboBox cb_errorFunc = new JFXComboBox(ol_errorFunc);
		Label lbl_maxItr = new Label("Max Nbr of Iterations");
		JFXTextField tf_maxItr = new JFXTextField();

		Label lbl_valFunc = new Label("Validation");
		ObservableList<String> ol_valFunc = FXCollections.observableArrayList("70-30", "Monte Carlo");
		JFXComboBox cb_valFunc = new JFXComboBox(ol_valFunc);

		// trHBox.getChildren().add(new Label("Training File"));
		JFXButton getFile = new JFXButton("Train On File");
		getFile.setStyle(cssButton);

		trngGrid.add(lbl_learnRate, 0, 0);
		trngGrid.add(tf_learnRate, 1, 0);
		trngGrid.setValignment(lbl_learnRate, VPos.BOTTOM);

		// trHBox.getChildren().add(lbl_learnRate);
		// trHBox.getChildren().add(tf_learnRate);
		// trVBox.getChildren().add(trHBox);
		// trHBox = new HBox(5);
		trngGrid.add(lbl_errorThresh, 0, 1);
		trngGrid.add(tf_errorThresh, 1, 1);
		trngGrid.setValignment(lbl_errorThresh, VPos.BOTTOM);

		// trHBox.getChildren().add(lbl_errorThresh);
		// trHBox.getChildren().add(tf_errorThresh);
		// trVBox.getChildren().add(trHBox);
		// trHBox = new HBox(5);

		trngGrid.add(lbl_errorFunc, 0, 2);
		trngGrid.add(cb_errorFunc, 1, 2);
		trngGrid.setValignment(lbl_errorFunc, VPos.BOTTOM);

		// trHBox.getChildren().add(lbl_errorFunc);
		// trHBox.getChildren().add(cb_errorFunc);
		// trVBox.getChildren().add(trHBox);
		// trHBox = new HBox(5);

		trngGrid.add(lbl_maxItr, 0, 3);
		trngGrid.add(tf_maxItr, 1, 3);
		trngGrid.setValignment(lbl_maxItr, VPos.BOTTOM);

		trngGrid.add(lbl_valFunc, 0, 4);
		trngGrid.add(cb_valFunc, 1, 4);
		trngGrid.setValignment(lbl_valFunc, VPos.BOTTOM);

		// trHBox.getChildren().add(lbl_maxItr);
		// trHBox.getChildren().add(tf_maxItr);
		// trVBox.getChildren().add(trHBox);
		// trHBox = new HBox(5);

		trngGrid.add(getFile, 0, 5);

		JFXButton error = new JFXButton("View Training Error");
		error.setStyle(cssButton);
		trngGrid.add(error, 0, 6);

		JFXButton viewPrec = new JFXButton("View Validation Error");
		viewPrec.setStyle(cssButton);
		trngGrid.add(viewPrec, 0, 7);

		error.setOnMouseClicked(event -> {
			Stage s = new Stage();
			Pane p = new Pane();
			final NumberAxis xAxis = new NumberAxis();
			final NumberAxis yAxis = new NumberAxis();
			LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
			XYChart.Series series = new XYChart.Series();
			series.setName("Error");

			for (int i = 0; i < neural.errorVals.size(); i++) {
				series.getData().add(new XYChart.Data(neural.trainingItrts.get(i), neural.errorVals.get(i)));

			}
			lineChart.getData().add(series);
			p.getChildren().add(lineChart);

			Scene scene = new Scene(p, p.getMaxWidth(), p.getMaxHeight());
			s.setScene(scene);
			s.setResizable(true);
			s.setTitle("Error");
			s.show();
		});

		ArrayList<Double> precisionError = new ArrayList<>();
		ArrayList<Double> precisionIndex = new ArrayList<>();
		getFile.setOnMouseClicked(event -> {
			try {
				FileChooser fileChooser = new FileChooser();
				File selectedFile = fileChooser.showOpenDialog(null);
				Path srcPath = selectedFile.toPath();
				Scanner scan = new Scanner(srcPath.toFile());
				int itrt = 0;
				ArrayList<Double> arrIn;
				// Counting number of training pairs
				double precision = 0.0;
				int countPair = 0;
				while (scan.hasNext()) {
					String line = scan.nextLine();
					if (line.startsWith("X")) {
						countPair++;
					}
				}

				if (cb_valFunc.getValue().equals("70-30")) {
					int trainStrt = 0;
					int trainEnd = (int) Math.round(0.7 * countPair);
					System.out.println("Train End: " + trainEnd);
					int validStrt = trainEnd + 1;
					int validEnd = countPair;
					scan = new Scanner(srcPath.toFile());
					itrt = 0;
					while (scan.hasNext()) {

						String line = scan.nextLine();
						Pattern pattern = Pattern.compile("[01]");
						Matcher match = pattern.matcher(line);
						if (itrt < trainEnd) {

							if (line.startsWith("X")) {
								int i = 0;
								while (match.find()) {
									inputsArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}
								if (itrt == 0) {
									neural.generateNetworkOutput(new Matrix(inputsArr2));
								}
							} else if (line.startsWith("D")) {
								int i = 0;
								while (match.find()) {
									desiredArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}
								neural.trainNetwork(new Matrix(inputsArr2), new Matrix(desiredArr2),
										Double.parseDouble(tf_learnRate.getText()),
										Double.parseDouble(tf_errorThresh.getText()), (String) cb_errorFunc.getValue(),
										Integer.parseInt(tf_maxItr.getText()));
								neural.generateNetworkOutput(new Matrix(inputsArr2)).print(5, 1);
								itrt++;
							}
						} else {

							if (line.startsWith("X")) {
								int i = 0;
								while (match.find()) {
									inputsArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}

							} else if (line.startsWith("D")) {
								int i = 0;
								while (match.find()) {
									desiredArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}

								Matrix desired = new Matrix(desiredArr2);
								Matrix actual = neural.generateNetworkOutput(new Matrix(inputsArr2));

								new Matrix(inputsArr2).print(5, 1);
								desired.print(5, 1);
								Matrix setError = desired.minus(actual);
								System.out.println("Error Set");
								setError.print(5, 1);
								double errorMSE = (Math.pow(setError.normF(), 2)
										/ (setError.getColumnDimension() * setError.getRowDimension()));
								precisionError.add(errorMSE);
								precisionIndex.add((double) (itrt - trainEnd));
								System.out.println(itrt);
								itrt++;
							}
						}

					}

					arrIn = toArrayList(inputsArr2);
					neural.drawNeuron(arrIn, pane, new Matrix(inputsArr2));
				}

				if (cb_valFunc.getValue().equals("Monte Carlo")) {
					SetCross s = new SetCross();
					int[] mntCrlIndex = s.monteCarlo(countPair);
					int strtVal = mntCrlIndex[0];
					int endVal = mntCrlIndex[1];
					System.out.println("Start Val: " + strtVal);
					System.out.println("End Val: " + endVal);
					scan = new Scanner(srcPath.toFile());
					itrt = 0;
					while (scan.hasNext()) {

						String line = scan.nextLine();
						Pattern pattern = Pattern.compile("[01]");
						Matcher match = pattern.matcher(line);
						if (itrt < strtVal || itrt > endVal) {
							if (line.startsWith("X")) {
								int i = 0;
								while (match.find()) {
									inputsArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}
								if (itrt == 0) {
									neural.generateNetworkOutput(new Matrix(inputsArr2));
								}
							} else if (line.startsWith("D")) {
								int i = 0;
								while (match.find()) {
									desiredArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}
								neural.trainNetwork(new Matrix(inputsArr2), new Matrix(desiredArr2),
										Double.parseDouble(tf_learnRate.getText()),
										Double.parseDouble(tf_errorThresh.getText()), (String) cb_errorFunc.getValue(),
										Integer.parseInt(tf_maxItr.getText()));
								neural.generateNetworkOutput(new Matrix(inputsArr2)).print(5, 1);
								itrt++;
							}
						}

					}
					scan = new Scanner(srcPath.toFile());
					itrt = strtVal;
					while (scan.hasNext() && itrt <= endVal) {

						String line = scan.nextLine();
						Pattern pattern = Pattern.compile("[01]");
						Matcher match = pattern.matcher(line);
						if (itrt >= strtVal && itrt <= endVal) {
							if (line.startsWith("X")) {
								int i = 0;
								while (match.find()) {
									inputsArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}
								if (itrt == 0) {
									neural.generateNetworkOutput(new Matrix(inputsArr2));
								}
							} else if (line.startsWith("D")) {
								int i = 0;
								while (match.find()) {
									desiredArr2[i][0] = Double.parseDouble(match.group());
									i++;
								}
								Matrix desired = new Matrix(desiredArr2);
								Matrix actual = neural.generateNetworkOutput(new Matrix(inputsArr2));
								Matrix setError = desired.minus(actual);
								double errorMSE = (Math.pow(setError.normF(), 2)
										/ (setError.getColumnDimension() * setError.getRowDimension()));
								precisionError.add(errorMSE);
								System.out.println("itrt-strtVal: " + (itrt - strtVal));
								precisionIndex.add((double) (itrt - strtVal));
								itrt++;
							}
						}

					}

					arrIn = toArrayList(inputsArr2);
					neural.drawNeuron(arrIn, pane, new Matrix(inputsArr2));
				}

			} catch (Exception exc) {
				exc.printStackTrace();
			}
		});

		viewPrec.setOnMouseClicked(event -> {
			Stage s = new Stage();
			Pane p = new Pane();
			final NumberAxis xAxis = new NumberAxis();
			final NumberAxis yAxis = new NumberAxis();
			LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
			XYChart.Series series = new XYChart.Series();
			series.setName("Validation error graph");

			for (int i = 0; i < precisionError.size(); i++) {
				series.getData().add(new XYChart.Data(precisionIndex.get(i), precisionError.get(i)));

			}
			lineChart.getData().add(series);
			p.getChildren().add(lineChart);

			Scene scene = new Scene(p, p.getMaxWidth(), p.getMaxHeight());
			s.setScene(scene);
			s.setResizable(true);
			s.setTitle("Validation error graph");
			s.show();
		});

		ntwrkScrl.setContent(pane);
		ntwrkScrl.setBackground(new Background(new BackgroundFill(Color.WHITE, CornerRadii.EMPTY, Insets.EMPTY)));
		// gp.setBackground(new Background(new BackgroundFill(Color.WHITE,
		// CornerRadii.EMPTY, Insets.EMPTY)));

		scrlPane2.setContent(trngGrid);
		scrlPane.setContent(initGrid);
		ttlPane.setContent(scrlPane);
		ttlPane2.setContent(scrlPane2);
		inPane.add(ttlPane, 0, 0);
		inPane.add(ttlPane2, 0, 1);
		gp.add(inPane, 0, 0);
		gp.add(ntwrkScrl, 1, 0);
		// gp.setPrefWidth(pane.getWidth());
		Scene scene = new Scene(gp, gp.getMaxWidth(), gp.getMaxHeight());
		stage.setScene(scene);
		// stage.setResizable(false);
		// stage.initStyle(StageStyle.UNDECORATED);
		Rectangle2D primScreen = Screen.getPrimary().getBounds();
		stage.setX(primScreen.getWidth() / 5);
		stage.setY(primScreen.getHeight() / 12);
		stage.setTitle("NN");
		stage.show();

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

	public static ArrayList<Double> toArrayList(double[][] arr) {
		ArrayList<Double> out = new ArrayList<Double>();
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[0].length; j++) {
				out.add(arr[i][j]);
			}
		}
		return out;
	}

	public static ArrayList<Layer> initializeNetwork(double[][] inputArr, ArrayList<double[][]> weightsArr,
			ArrayList<ArrayList<Double>> biases, ArrayList<ArrayList<String>> actFuncs) {
		Matrix input = new Matrix(inputArr);
		ArrayList<Matrix> weights = new ArrayList<>();
		ArrayList<Layer> layers = new ArrayList<>();

		for (int i = 0; i < weightsArr.size(); i++)
			weights.add(new Matrix(weightsArr.get(i)));

		for (int i = 0; i < weightsArr.size(); i++) {
			Matrix layerWeights = weights.get(i);
			int neuronsCount = weights.get(i).getColumnDimension();
			System.out.println(neuronsCount);
			layerWeights.print(5, 1);
			ArrayList<Neuron> layerNeurons = new ArrayList<>();
			if (i > 0)
				layers.get(i - 1).generateOutput();
			for (int j = 0; j < neuronsCount; j++) {
				if (i == 0) {
					Neuron N = new Neuron(input, layerWeights, biases.get(i).get(j), actFuncs.get(i).get(j));
					layerNeurons.add(N);
				} else {
					Neuron N = new Neuron(layers.get(i - 1).layerOutput, layerWeights, biases.get(i).get(j),
							actFuncs.get(i).get(j));
					layerNeurons.add(N);
				}
			}
			if (i == 0)
				layers.add(new Layer(input, layerWeights, layerNeurons));
			else {
				layers.get(i - 1).generateOutput();
				layers.add(new Layer(layers.get(i - 1).layerOutput, layerWeights, layerNeurons));
			}
		}
		layers.get(layers.size() - 1).generateOutput();
		return layers;
	}

}