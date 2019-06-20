#!/usr/bin/env node
const {
	calculateError,
	deltaCalculator,
	forwardPropagator,
	sigmoid,
	sigmoidDerivative,
	updateSynapse,
} = require("./utils");

// prettier-ignore
const X = [
	[0, 0, 1],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 1],
];

// prettier-ignore
const y = [
	[0],
	[0],
	[1],
	[1],
];

const iterations = process.env.ITERATIONS
	? parseInt(process.env.ITERATIONS, 10)
	: 100000;

const forwardPropagate = forwardPropagator(sigmoid);
const calculateDelta = deltaCalculator(sigmoidDerivative);

const initialState = {
	layer0: X,
	synapse0: [...new Array(3)].map(() => [2 * Math.random() - 1]),
	layer1: null,
};

const reducer = ({ layer0, synapse0 }, _, i) => {
	const layer1 = forwardPropagate(layer0, synapse0);
	const layer1Error = calculateError(y, layer1);

	if (process.env.DEBUG && (i + 1) % (iterations / 10) === 0) {
		console.info(`Error after ${i + 1} iterations:\n`, layer1Error);
	}

	const layer1Delta = calculateDelta(layer1, layer1Error);
	const updatedSynapse = updateSynapse(layer0, synapse0, layer1Delta);

	return {
		layer0,
		synapse0: updatedSynapse,
		layer1,
	};
};

const { layer1 } = [...new Array(iterations)].reduce(reducer, initialState);

console.info("Output after training:");
console.info(layer1);

/* E.g.
[ 0.003017743106269471,
  0.0024610083284618603,
  0.9979917093392362,
  0.9975371365031968 ]

	Time 0.924s
*/
