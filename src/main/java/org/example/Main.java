package org.example;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedList;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;

import java.util.Vector;

public class Main extends AbstractClassifier {

	protected double clonalFactor;
	protected int antibodyPoolSize;
	protected int selectionPoolSize;
	protected int totalReplacement;
	protected int numGenerations;
	protected long seed;
	protected double remainderPoolRatio;

	protected CLONALGAlgorithm algorithm;

	private final static String[] PARAMETERS = {"B", "N", "n", "D", "G", "S", "R"};

	private final static String[] DESCRIPTIONS =
			{
					"Clonal factor (beta). Used to scale the number of clones created by the selected best antibodies.",
					"Antibody pool size (N). The total antibodies maintained in the memory pool and remainder pool.",
					"Selection pool size (n). The total number of best antibodies selected for cloning and mutation each iteration.",
					"Total replacements (d). The total number of antibodies in the remainder pool that are replaced each iteration. Typically 5%-8%",
					"Total generations. The total number of times that all antigens are exposed to the system.",
					"Random number generator seed. Seed used to initialise the random number generator.",
					"Remainder pool percentage. The percentage of the total antibody pool size allocated for the remainder pool."
			};

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		result.disableAll();
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.setMinimumNumberInstances(1);

		return result;
	}

	public void buildClassifier(Instances data) throws Exception {
		Instances trainingInstances = new Instances(data);
		trainingInstances.deleteWithMissingClass();

		getCapabilities().testWithFail(trainingInstances);

		algorithm = new CLONALGAlgorithm(clonalFactor, antibodyPoolSize, selectionPoolSize, totalReplacement, numGenerations, seed, remainderPoolRatio);

		algorithm.train(trainingInstances);
	}

	public double classifyInstance(Instance instance) throws Exception {
		if (algorithm == null) {
			throw new Exception("Algorithm has not been prepared.");
		}
		return algorithm.classify(instance);
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer(1000);
		buffer.append("CLONALG v1.0.");
		return buffer.toString();
	}

	public Enumeration listOptions() {
		Vector<Option> list = new Vector<>(15);

		Enumeration e = super.listOptions();
		while (e.hasMoreElements()) {
			list.add((Option) e.nextElement());
		}

		for (int i = 0; i < PARAMETERS.length; i++) {
			Option o = new Option(DESCRIPTIONS[i], PARAMETERS[i], 1, "-" + PARAMETERS[i]);
			list.add(o);
		}

		return list.elements();
	}

	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);

		this.clonalFactor = 0.2;
		this.antibodyPoolSize = 100;
		this.selectionPoolSize = 10;
		this.totalReplacement = 5;
		this.numGenerations = 100;
		this.seed = 1;
		this.remainderPoolRatio = 0.2;
	}

	public static void main(String[] args) {
		runClassifier(new Main(), args);
	}
}