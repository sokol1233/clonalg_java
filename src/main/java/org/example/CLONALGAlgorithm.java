/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.example;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class CLONALGAlgorithm implements Serializable {

	protected final double clonalFactor;

	protected final int antibodyPoolSize;

	protected final int selectionPoolSize;

	protected final int replacementPoolSize;

	protected final int numGenerations;

	protected final long seed;

	protected final double remainderPoolRatio;

	protected Antibody[] memoryPool;

	protected Antibody[] remainderPool;

	protected Random rand;

	protected DistanceFunction affinityFunction;

	public CLONALGAlgorithm(double aClonalFactor, int aAntibodyPoolSize, int aSelectionPoolSize, int aReplacementPoolSize, int aNumGenerations,
			long aSeed, double aRemainderPoolRatio) {
		clonalFactor = aClonalFactor;
		antibodyPoolSize = aAntibodyPoolSize;
		selectionPoolSize = aSelectionPoolSize;
		replacementPoolSize = aReplacementPoolSize;
		numGenerations = aNumGenerations;
		seed = aSeed;
		remainderPoolRatio = aRemainderPoolRatio;
	}


	protected void algorithmSetup(Instances aAntigens) {
		rand = new Random(seed);
		int remainderSize = (int) Math.round(antibodyPoolSize * remainderPoolRatio);
		int memorySize = (antibodyPoolSize - remainderSize);

		if (remainderSize == 0) {
			throw new RuntimeException("Remainder pool size cannot be zero!");
		}

		if (memorySize == 0) {
			throw new RuntimeException("Memory pool size cannot be zero!");
		}

		if (remainderSize < replacementPoolSize) {
			throw new RuntimeException(String.format("The size of the remainder pool [%d] is less than the number of elements replaced each iteration [%d].",
					remainderSize, replacementPoolSize));
		}

		memoryPool = new Antibody[memorySize];
		remainderPool = new Antibody[remainderSize];
		affinityFunction = new DistanceFunction(aAntigens);
	}

	protected void initialiseAntibodyPool(Instances aAntigens) {
		aAntigens.randomize(rand);

		for (int i = 0; i < remainderPool.length; i++) {
			remainderPool[i] = new Antibody(aAntigens.instance(rand.nextInt(aAntigens.numInstances())));
		}

		for (int i = 0; i < memoryPool.length; i++) {
			memoryPool[i] = new Antibody(aAntigens.instance(rand.nextInt(aAntigens.numInstances())));
		}
	}

	protected void train(Instances aAntigens) {
		algorithmSetup(aAntigens);

		initialiseAntibodyPool(aAntigens);

		for (int i = 0; i < numGenerations; i++) {
			aAntigens.randomize(rand);

			for (int j = 0; j < aAntigens.numInstances(); j++) {
				Instance currentInstance = aAntigens.instance(j);

				calculateAffinity(remainderPool, currentInstance);
				calculateAffinity(memoryPool, currentInstance);

				Antibody[] bestSet = selectBestAntibodySet();

				Antibody[] cloneSet = prepareCloneSet(bestSet, currentInstance);

				calculateAffinity(cloneSet, currentInstance);

				Arrays.sort(cloneSet);
				Antibody candidate = cloneSet[0];

				if (candidate.getClassification() == currentInstance.classValue()) {
					Arrays.sort(memoryPool);
					if (candidate.getAffinity() < memoryPool[0].getAffinity()) {
						memoryPool[0] = candidate;
					}
				}

				Arrays.sort(remainderPool);
				for (int k = cloneSet.length - 1; k < replacementPoolSize; k++) {
					remainderPool[k] = generateRandomAntibodyInRange(remainderPool[k], currentInstance);
				}
			}
		}
	}


	public double classify(Instance aInstance) {
		calculateAffinity(memoryPool, aInstance);
		Arrays.sort(memoryPool);
		return memoryPool[0].getClassification();
	}


	protected Antibody generateRandomAntibodyInRange(Antibody aAntibody, Instance aInstance) {
		mutateClone(aAntibody, 1.0, aInstance);
		double[] data = aAntibody.getAttributes();
		data[aAntibody.getClassIndex()] = rand.nextInt(aInstance.classAttribute().numValues());
		return aAntibody;
	}

	protected void mutateClone(Antibody aClone, double aMutationRate, Instance aInstance) {
		double[][] minmax = affinityFunction.getMinMax();
		AttributeDistance[] attribs = affinityFunction.getDistanceMeasures();

		double[] data = aClone.getAttributes();

		for (int i = 0; i < data.length; i++) {
			if (attribs[i].isClass()) {
				continue;
			} else if (attribs[i].isNominal()) {
				data[i] = rand.nextInt(aInstance.attribute(i).numValues());
			} else if (attribs[i].isNumeric()) {
				double range = (minmax[i][1] - minmax[i][0]);
				range = (range * aMutationRate);

				double min = Math.max(data[i] - (range / 2.0), minmax[i][0]);
				double max = Math.min(data[i] + (range / 2.0), minmax[i][1]);

				data[i] = min + (rand.nextDouble() * (max - min));
			} else {
				throw new RuntimeException("Unsupported attribute type!");
			}
		}
	}

	protected Antibody[] prepareCloneSet(Antibody[] aBestSet, Instance aInstance) {
		LinkedList<Antibody> clones = new LinkedList<>();

		for (int i = 1; i <= aBestSet.length; i++) {
			int numClones = (int) Math.round((clonalFactor * antibodyPoolSize) / i);

			Antibody current = aBestSet[i - 1];
			double mutationRate = (double) i / (double) aBestSet.length;

			for (int j = 0; j < numClones; j++) {
				Antibody a = new Antibody(current);
				mutateClone(a, mutationRate, aInstance);
				clones.add(a);
			}
		}

		return clones.toArray(new Antibody[0]);
	}


	protected Antibody[] selectBestAntibodySet() {
		Antibody[] bestSet = new Antibody[selectionPoolSize];
		LinkedList<Antibody> totalSet = new LinkedList<>();

		for (int i = 0; i < remainderPool.length; i++) {
			totalSet.add(remainderPool[i]);
		}
		for (int i = 0; i < memoryPool.length; i++) {
			totalSet.add(memoryPool[i]);
		}

		Collections.sort(totalSet);

		for (int i = 0; i < bestSet.length; i++) {
			bestSet[i] = totalSet.get(i);
		}

		return bestSet;
	}

	protected void calculateAffinity(Antibody[] antibodies, Instance aInstance) {
		double[] data = aInstance.toDoubleArray();

		for (int i = 0; i < antibodies.length; i++) {
			double affinity = affinityFunction.calculateDistance(antibodies[i].getAttributes(), data);
			antibodies[i].setAffinity(affinity);
		}
	}
}
