const elementwiseOperation = fn => (array1, array2) =>
	array1.map((n, i) => fn(n, array2[i]));

const elementwiseAdd = elementwiseOperation((a, b) => a + b);
const elementwiseSubtract = elementwiseOperation((a, b) => a - b);
const elementwiseMultiply = elementwiseOperation((a, b) => a * b);

const flatten = array =>
	array.reduce((previous, next) => previous.concat(next), []);

const dotProduct = (array1, array2) =>
	elementwiseMultiply(array1, array2).reduce((a, b) => a + b, 0);

const transpose = matrix => matrix[0].map((_, i) => matrix.map(row => row[i]));

const matrixMultiply = (matrix1, matrix2) => {
	const transposedMatrix2 = transpose(matrix2);
	return matrix1.map(row =>
		transposedMatrix2.map(column => dotProduct(row, column)),
	);
};

const sigmoid = n => 1 / (1 + Math.exp(-n));
const sigmoidDerivative = n => n * (1 - n);

module.exports = {
	dotProduct,
	elementwiseAdd,
	elementwiseOperation,
	elementwiseMultiply,
	elementwiseSubtract,
	flatten,
	matrixMultiply,
	sigmoid,
	sigmoidDerivative,
	transpose,
};
