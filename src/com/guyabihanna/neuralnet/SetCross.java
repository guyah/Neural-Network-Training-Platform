package com.guyabihanna.neuralnet;

import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class SetCross {

	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub

		Path path = Paths.get("inNeuron.txt");
		Scanner scan = new Scanner(path.toFile());
		int countPair = 0;

		while (scan.hasNext()) {
			String line = scan.nextLine();
			if (line.startsWith("X")) {
				countPair++;
			}
		}

		scan.close();

		int trainSt = 0;
		int trainEnd = (int) Math.round(0.7 * countPair);
		int validSt = trainEnd + 1;
		int validEnd = countPair;

		System.out.println(countPair);
	//	System.out.println(trainSt);
	//	System.out.println(trainEnd);
	//	System.out.println(validSt);
	//	System.out.println(validEnd);

		System.out.println(Arrays.toString(monteCarlo(countPair)));// return validation start and end

	}

	public static int[] randomCut(int total, int s) {
		int[] arr = new int[2];
		Random rmd = new Random();
		int a = rmd.nextInt(total);
		int b = a - s;
		while (b < 0) {
			a = rmd.nextInt(total);
			b = a - s;
		}
		arr[0] = b;
		arr[1] = a;
		return arr;
	}

	public static int[] monteCarlo(int total) {
		int[] arr = new int[2];
		Random rmd = new Random();
		int a = rmd.nextInt(total);
		int s = rmd.nextInt(total - total / 2);
		int b = a - s;
		while (b < 0 || b == a) {
			a = rmd.nextInt(total);
			b = a - s;
			s = rmd.nextInt(total - total / 2);
		}
		arr[0] = b;
		arr[1] = a;
		return arr;
	}

}