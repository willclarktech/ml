const elementwiseOperation = fn => (array1, array2) =>
	array1.map((n, i) => fn(n, array2[i]));

const elementwiseAdd = elementwiseOperation((a, b) => a + b);
const elementwiseSubtract = elementwiseOperation((a, b) => a - b);
const elementwiseMultiply = elementwiseOperation((a, b) => a * b);

const flatten = array =>
	array.reduce((previous, next) => previous.concat(next), []);

const unflatten = array => array.map(n => [n]);

const deepMap = (fn, array) => array.map(row => row.map(fn));

const dotProduct = (array1, array2) =>
	elementwiseMultiply(array1, array2).reduce((a, b) => a + b, 0);

const transpose = matrix => matrix[0].map((_, i) => matrix.map(row => row[i]));

const matrixMultiply = (matrix1, matrix2) => {
	const transposedMatrix2 = transpose(matrix2);
	return matrix1.map(row =>
		transposedMatrix2.map(column => dotProduct(row, column)),
	);
};

const calculateError = (expected, actual) =>
	unflatten(elementwiseSubtract(flatten(expected), flatten(actual)));

const forwardPropagator = nonlinearFn => (layer, synapse) =>
	deepMap(nonlinearFn, matrixMultiply(layer, synapse));

const deltaCalculator = derivativeFn => (layer, error) => {
	const derivatives = deepMap(derivativeFn, layer);
	return elementwiseMultiply(flatten(error), flatten(derivatives));
};

const calculateUpdate = (layer, delta) =>
	matrixMultiply(transpose(layer), unflatten(delta));

const updateSynapse = (layer, synapse, delta) => {
	const update = calculateUpdate(layer, delta);
	return elementwiseOperation(elementwiseAdd)(synapse, update);
};

const sigmoid = n => 1 / (1 + Math.exp(-n));
const sigmoidDerivative = n => n * (1 - n);

module.exports = {
	calculateError,
	calculateUpdate,
	deltaCalculator,
	dotProduct,
	elementwiseAdd,
	elementwiseOperation,
	elementwiseMultiply,
	elementwiseSubtract,
	flatten,
	forwardPropagator,
	matrixMultiply,
	sigmoid,
	sigmoidDerivative,
	transpose,
	updateSynapse,
};
