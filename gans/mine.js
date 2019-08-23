const {
	bceLoss,
	boxMueller,
	sigmoid,
	stats,
	vectorMatrixMultiply,
} = require("./utils");

// Data parameters

dataMean = 4;
dataStandardDeviation = 1.25;

// Model parameters

gParameters = {
	inputSize: 1,
	hiddenSize: 5,
	outputSize: 1,
	learningRate: 1e-2,
	sgdMomentum: 0.9,
	activationFunction: Math.tanh,
};

dParameters = {
	inputSize: 500,
	hiddenSize: 10,
	outputSize: 1,
	learningRate: 1e-3,
	sgdMomentum: 0.9,
	activationFunction: sigmoid,
};

// Training parameters

numEpochs = 1;
dStepsPerEpoch = 1;
gStepsPerEpoch = 1;

// Distribution functions

const createDistributionSampler = (mu, sigma) => n =>
	new Array(n).fill().map(boxMueller.bind(null, mu, sigma));

const sampleDistribution = createDistributionSampler(
	dataMean,
	dataStandardDeviation,
);

const createGeneratorInputSampler = generatorInputSize => n =>
	new Array(n)
		.fill()
		.map(() => new Array(generatorInputSize).fill().map(Math.random));

const sampleGeneratorInput = createGeneratorInputSampler(gParameters.inputSize);

const calculateError = bceLoss;

// Models

class Generator {
	constructor({
		inputSize,
		hiddenSize,
		outputSize,
		learningRate,
		sgdMomentum,
		activationFunction,
	}) {
		this.activationFunction = activationFunction;

		this.weights = [
			new Array(hiddenSize)
				.fill()
				.map(() => new Array(inputSize).fill().map(Math.random)),
			new Array(hiddenSize)
				.fill()
				.map(() => new Array(hiddenSize).fill().map(Math.random)),
			new Array(outputSize)
				.fill()
				.map(() => new Array(hiddenSize).fill().map(Math.random)),
		];
		console.log("G WEIGHTS", this.weights);
	}

	forward(input) {
		const layer1 = vectorMatrixMultiply(input, this.weights[0]).map(
			this.activationFunction,
		);
		const layer2 = vectorMatrixMultiply(layer1, this.weights[1]).map(
			this.activationFunction,
		);
		return vectorMatrixMultiply(layer2, this.weights[2]);
	}
}

class Discriminator {
	constructor({
		inputSize,
		hiddenSize,
		outputSize,
		learningRate,
		sgdMomentum,
		activationFunction,
	}) {
		this.inputSize = inputSize;
		this.activationFunction = activationFunction;

		this.weights = [
			new Array(hiddenSize)
				.fill()
				.map(() => new Array(inputSize).fill().map(Math.random)),
			new Array(hiddenSize)
				.fill()
				.map(() => new Array(hiddenSize).fill().map(Math.random)),
			new Array(outputSize)
				.fill()
				.map(() => new Array(hiddenSize).fill().map(Math.random)),
		];
	}

	forward(input) {
		const layer1 = vectorMatrixMultiply(input, this.weights[0]).map(
			this.activationFunction,
		);
		const layer2 = vectorMatrixMultiply(layer1, this.weights[1]).map(
			this.activationFunction,
		);
		return vectorMatrixMultiply(layer2, this.weights[2]).map(
			this.activationFunction,
		);
	}
}

// Training functions

const trainD = (d, g) => {
	const dRealData = sampleDistribution(d.inputSize);
	console.log("REAL");
	console.log(dRealData);
	const dRealDecision = d.forward(dRealData);
	console.log("REAL DECISION", dRealDecision);
	const dRealError = calculateError(dRealDecision, 1);

	const dGeneratorInput = sampleGeneratorInput(d.inputSize);
	// console.log("GENERATOR INPUT");
	// console.log(dGeneratorInput);
	const dFakeData = dGeneratorInput.map(g.forward.bind(g));
	// console.log("FAKE");
	// console.log(dFakeData);
	const dFakeDecision = d.forward(dFakeData);
	const dFakeError = calculateError(dFakeDecision, 0);

	return {
		dRealError,
		dFakeError,
		dRealData,
		dFakeData,
	};
};

const trainG = (d, g) => {
	const gError = [];
	return {
		gError,
	};
};

// Print function

const printProgress = (
	epoch,
	dRealError,
	dFakeError,
	gError,
	dRealData,
	dFakeData,
) => {
	const dRealDataStats = stats(dRealData);
	const dFakeDataStats = stats(dFakeData);
	const dRealDataString = `${dRealDataStats.mean} ${dRealDataStats.std}`;
	const dFakeDataString = `${dFakeDataStats.mean} ${dFakeDataStats.std}`;

	console.info(`Epoch ${epoch}:`);
	console.info(
		`D (${dRealError[0]} real error, ${dFakeError[0]} fake error) G (${
			gError[0]
		} error)`,
	);
	console.info(`Real dist ${dRealDataString}, Fake dist ${dFakeDataString}`);
	console.info();
};

// Main training loop

const train = (epochs, dSteps, gSteps, printInterval = 100) => {
	const g = new Generator(gParameters);
	const d = new Discriminator(dParameters);

	let dRealError;
	let dFakeError;
	let gError;
	let dRealData;
	let dFakeData;

	for (let epoch = 0; epoch < epochs; ++epoch) {
		for (let dStep = 0; dStep < dSteps; ++dStep) {
			({ dRealError, dFakeError, dRealData, dFakeData } = trainD(d, g));
		}

		for (let gStep = 0; gStep < gSteps; ++gStep) {
			({ gError } = trainG(d, g));
		}

		if (epoch % printInterval === 0) {
			printProgress(
				epoch,
				dRealError,
				dFakeError,
				gError,
				dRealData,
				dFakeData,
			);
		}
	}
};

train(numEpochs, dStepsPerEpoch, gStepsPerEpoch);
