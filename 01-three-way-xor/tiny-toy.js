#!/usr/bin/env node
const {
	elementwiseAdd,
	elementwiseMultiply,
	elementwiseSubtract,
	flatten,
	matrixMultiply,
	sigmoid,
	sigmoidDerivative,
	transpose,
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

const layer0 = X;
let synapse0 = [...new Array(3)].map(() => [2 * Math.random() - 1]);
let layer1;
let layer1_error;
let layer1_delta;
let update;

const iterations = process.env.ITERATIONS
	? parseInt(process.env.ITERATIONS, 10)
	: 100000;

for (let i = 0; i < iterations; ++i) {
	layer1 = matrixMultiply(layer0, synapse0).map(([n]) => [sigmoid(n)]);
	layer1_error = elementwiseSubtract(y, flatten(layer1));

	if (process.env.DEBUG && i % (iterations / 10) === 0) {
		console.info(layer1_error);
	}

	layer1_delta = elementwiseMultiply(
		layer1_error,
		layer1.map(([n]) => [sigmoidDerivative(n)]),
	);
	update = matrixMultiply(transpose(layer0), layer1_delta.map(n => [n]));
	synapse0 = synapse0.map((v, i) => elementwiseAdd(v, update[i]));
}

console.info("Output after training:");
console.info(layer1);

/* E.g.
[ 0.003017743106269471,
  0.0024610083284618603,
  0.9979917093392362,
  0.9975371365031968 ]

	Time 0.677s
*/
