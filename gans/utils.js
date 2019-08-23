const sum = v => v.reduce((total, n) => total + n, 0);

const mean = distribution => sum(distribution) / distribution.length;

const standard_deviation = distribution => {
	const m = mean(distribution);
	return Math.sqrt(
		distribution.reduce((sum, n) => sum + (n - m) ** 2, 0) /
			distribution.length -
			1,
	);
};

const stats = distribution => ({
	mean: mean(distribution),
	std: standard_deviation(distribution),
});

const bceLoss = (actual, expected) => {
	if (![0, 1].includes(expected)) {
		throw new Error("Cannot calculate BCE loss on non-binary value");
	}

	return expected ? -Math.log(actual) : -Math.log(1 - actual);
};

const sigmoid = n => 1 / (1 + Math.exp(-n));

const boxMueller = (mu, sigma) => {
	const u1 = Math.random();
	const u2 = Math.random();
	const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
	return z0 * sigma + mu;
};

const vectorMatrixMultiply = (v, m) => {
	if (v.length !== m[0].length) {
		throw new Error(
			`Cannot multiply vector of length ${v.length} with a matrix with ${
				m[0].length
			} column(s)`,
		);
	}
	return m.map(row => sum(row.map((n, i) => n * v[i])));
};

module.exports = {
	bceLoss,
	boxMueller,
	sigmoid,
	stats,
	vectorMatrixMultiply,
};
