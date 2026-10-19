// Browser-native core of neural-architecture-generator.
// This is the donor's local lightweight architecture/modulation runtime, not
// a claim that each family is a full production implementation. Family
// transforms are small executable mathematical surrogates with trainable
// output biases; their purpose here is compositional modulation.

export const ARCHITECTURE_FAMILY_IDS = Object.freeze([
	"mlp",
	"conv1d-net",
	"rnn-net",
	"attention-net",
	"deep-feedforward",
	"highway",
	"resnet",
	"densenet",
	"cnn2d",
	"cnn3d",
	"unet",
	"efficientnet",
	"mobilenet",
	"convnext",
	"capsule",
	"vanilla-rnn",
	"bilstm",
	"transformer",
	"encoder-transformer",
	"decoder-transformer",
	"encoder-decoder",
	"vit",
	"swin",
	"perceiver",
	"gcn",
	"gat",
	"graphsage",
	"mpnn",
	"vae",
	"gan",
	"diffusion",
	"moe",
	"siamese",
	"nerf",
	"dqn",
	"hypernetwork",
]);

const FAMILY_META = Object.freeze({
	mlp: ["Multilayer Perceptron", "Feedforward"],
	"conv1d-net": ["1D Convolutional Network", "Convolution"],
	"rnn-net": ["Gated Recurrent Network", "Recurrent"],
	"attention-net": ["Self-Attention Network", "Attention"],
	"deep-feedforward": ["Deep Feedforward Network", "Feedforward"],
	highway: ["Highway Network", "Feedforward"],
	resnet: ["Residual Network", "Convolution"],
	densenet: ["DenseNet", "Convolution"],
	cnn2d: ["2D CNN", "Convolution"],
	cnn3d: ["3D CNN", "Convolution"],
	unet: ["U-Net", "Convolution"],
	efficientnet: ["EfficientNet", "Convolution"],
	mobilenet: ["MobileNet", "Convolution"],
	convnext: ["ConvNeXt", "Convolution"],
	capsule: ["Capsule Network", "Convolution"],
	"vanilla-rnn": ["Vanilla RNN", "Recurrent"],
	bilstm: ["Bidirectional LSTM", "Recurrent"],
	transformer: ["Transformer", "Attention"],
	"encoder-transformer": ["Encoder-only Transformer", "Attention"],
	"decoder-transformer": ["Decoder-only Transformer", "Attention"],
	"encoder-decoder": ["Encoder-Decoder Transformer", "Attention"],
	vit: ["Vision Transformer", "Attention"],
	swin: ["Swin Transformer", "Attention"],
	perceiver: ["Perceiver", "Attention"],
	gcn: ["Graph Convolutional Network", "Graph"],
	gat: ["Graph Attention Network", "Graph"],
	graphsage: ["GraphSAGE", "Graph"],
	mpnn: ["Message Passing Neural Network", "Graph"],
	vae: ["Variational Autoencoder", "Generative"],
	gan: ["Generative Adversarial Network", "Generative"],
	diffusion: ["Diffusion Model", "Generative"],
	moe: ["Mixture of Experts", "Routing"],
	siamese: ["Siamese Network", "Comparison"],
	nerf: ["Neural Radiance Field", "Implicit"],
	dqn: ["Deep Q-Network", "Reinforcement"],
	hypernetwork: ["Hypernetwork", "Meta"],
});

export const ARCHITECTURE_FAMILIES = Object.freeze(
	ARCHITECTURE_FAMILY_IDS.map((id) =>
		Object.freeze({
			id,
			name: FAMILY_META[id][0],
			group: FAMILY_META[id][1],
			runtime: "functional",
		}),
	),
);

const DEMO_SAMPLES = Object.freeze([
	Object.freeze({ input: [0.1, 0.2, 0.1, 0.2], label: 0 }),
	Object.freeze({ input: [0.1, 0.4, 0.7, 0.9], label: 1 }),
	Object.freeze({ input: [0.9, 0.7, 0.8, 0.6], label: 2 }),
	Object.freeze({ input: [0.45, 0.5, 0.48, 0.52], label: 1 }),
]);

const learnedBiases = new Map();

function biasFor(id) {
	if (!learnedBiases.has(id)) learnedBiases.set(id, [0, 0, 0]);
	return learnedBiases.get(id);
}

function relu(value) {
	return Math.max(0, value);
}

function dot(left, right) {
	return left.reduce(
		(sum, value, index) => sum + value * (right[index] ?? 0),
		0,
	);
}

function softmax(values) {
	const maximum = Math.max(...values);
	const exponentials = values.map((value) => Math.exp(value - maximum));
	const total = exponentials.reduce((sum, value) => sum + value, 0) || 1;
	return exponentials.map((value) => value / total);
}

function argmax(values) {
	return values.reduce(
		(best, value, index) => value > values[best] ? index : best,
		0,
	);
}

function adaptToStandardPort(tensor = []) {
	if (tensor.length === 4) return tensor.slice();
	if (!tensor.length) return [0, 0, 0, 0];
	const output = [0, 0, 0, 0];
	const divisor = Math.ceil(tensor.length / 4);
	tensor.forEach((value, index) => {
		output[index % 4] += Number(value || 0) / divisor;
	});
	return output;
}

const MLP_1 = Object.freeze([
	[0.4, 0.2, -0.3, 0.5],
	[-0.2, 0.6, 0.4, -0.1],
	[0.3, -0.5, 0.7, 0.2],
	[0.5, 0.1, 0.2, 0.6],
	[-0.4, 0.3, 0.1, 0.4],
	[0.2, 0.7, -0.2, 0.2],
	[0.6, -0.1, 0.4, -0.4],
	[-0.1, 0.5, 0.5, 0.3],
]);

const MLP_2 = Object.freeze([
	[0.6, -0.2, 0.4, 0.1, -0.3, 0.5, 0.2, -0.1],
	[-0.1, 0.6, 0.2, -0.4, 0.5, 0.1, -0.3, 0.4],
	[0.2, 0.1, 0.6, 0.4, -0.2, -0.1, 0.5, 0.3],
]);

function dense(input, weights, bias, activate = true) {
	return weights.map((row, index) => {
		const value = dot(row, input) + (bias[index] ?? 0);
		return activate ? relu(value) : value;
	});
}

function runMLP(input) {
	const hidden = dense(
		input,
		MLP_1,
		[0.1, 0.05, 0.1, 0, 0.1, 0, 0.05, 0.1],
	);
	return dense(hidden, MLP_2, [0.1, -0.05, 0.05], false);
}

function runConv(input) {
	const windows = [
		input.slice(0, 2),
		input.slice(1, 3),
		input.slice(2, 4),
	];
	const features = windows.map((window) => relu(dot(window, [0.7, -0.4]) + 0.2));
	return [
		features[0] + features[1],
		features[1] + features[2],
		features[0] + features[2],
	].map((value, index) => value * [0.9, 1.05, 0.8][index] - 0.15);
}

function runRNN(input) {
	let hidden = [0.1, -0.05, 0.08, 0.02, -0.04, 0.06];
	for (const value of input) {
		const previous = hidden;
		hidden = previous.map((state, index) =>
			Math.tanh(
				state * 0.65 +
				value * (0.12 + index * 0.03) +
				(previous[(index + 1) % previous.length] ?? 0) * 0.08,
			),
		);
	}
	return [
		dot(hidden, [0.3, -0.2, 0.4, 0.1, 0.2, -0.1]),
		dot(hidden, [-0.2, 0.4, 0.1, 0.3, -0.2, 0.2]),
		dot(hidden, [0.1, 0.2, -0.3, 0.4, 0.1, 0.3]),
	];
}

function runAttention(input) {
	const attention = softmax(
		input.map((value, index) => value * 1.5 + index * 0.04),
	);
	const context = dot(attention, input);
	return [
		context * 0.8 + input[0] * 0.2,
		context * 0.7 + input[2] * 0.3,
		context * 0.6 + input[3] * 0.4,
	];
}

function affine3(input, matrix, bias = [0, 0, 0]) {
	return matrix.map((row, index) => dot(row, input) + bias[index]);
}

function familyDense(input, seed, activate = true) {
	return affine3(
		input,
		[
			[0.15 + seed * 0.01, -0.08, 0.11],
			[-0.12, 0.17 + seed * 0.008, 0.06],
			[0.09, 0.04, 0.13 + seed * 0.006],
		],
		[seed * 0.01, -seed * 0.005, seed * 0.003],
	).map((value) => activate ? Math.tanh(value) : value);
}

function familyStack(input, seed) {
	return familyDense(familyDense(input, seed, true), seed + 2, true);
}

function residual3(input, seed) {
	return familyDense(input, seed).map(
		(value, index) => value + (input[index % input.length] ?? 0) * 0.25,
	);
}

function configuredInput(input, options) {
	const numeric = Object.values(options || {}).reduce(
		(sum, value) =>
			sum + (typeof value === "number" ? value : String(value).length * 0.1),
		0,
	);
	return adaptToStandardPort(input).map(
		(value, index) =>
			value * (1 + Math.tanh(numeric * 0.01)) +
			numeric * 0.0007 * (index + 1),
	);
}

function familyLogits(id, rawInput, options = {}) {
	const input = configuredInput(rawInput, options);
	switch (id) {
		case "mlp":
			return { logits: runMLP(input), parameters: 67 };
		case "conv1d-net":
			return { logits: runConv(input), parameters: 27 };
		case "rnn-net":
		case "vanilla-rnn":
			return {
				logits: runRNN(
					id === "vanilla-rnn" ? input.map((value) => value * 0.8) : input,
				),
				parameters: id === "vanilla-rnn" ? 56 : 81,
			};
		case "attention-net":
			return { logits: runAttention(input), parameters: 48 };
		case "deep-feedforward":
			return { logits: familyStack(input, 2), parameters: 132 };
		case "highway": {
			const gate = softmax([input[0], input[1], input[2] ?? 0]);
			const transformed = familyDense(input, 4);
			return {
				logits: transformed.map(
					(value, index) =>
						value * gate[index] + (input[index] ?? 0) * (1 - gate[index]),
				),
				parameters: 148,
			};
		}
		case "resnet":
			return { logits: residual3(input, 5), parameters: 186 };
		case "densenet": {
			const first = familyDense(input, 6);
			return {
				logits: familyDense([...input, ...first].slice(0, 4), 8),
				parameters: 214,
			};
		}
		case "cnn2d":
			return {
				logits: runConv(
					input.map((value, index) => value * (index % 2 ? 0.85 : 1.1)),
				),
				parameters: 96,
			};
		case "cnn3d":
			return {
				logits: runConv([
					input[0],
					(input[1] + input[2]) / 2,
					(input[2] + input[3]) / 2,
					input[3],
				]),
				parameters: 144,
			};
		case "unet": {
			const encoded = familyDense(input, 10);
			return {
				logits: encoded.map(
					(value, index) =>
						value + (input[(index + 1) % input.length] ?? 0) * 0.18,
				),
				parameters: 292,
			};
		}
		case "efficientnet":
			return {
				logits: familyDense(input.map((value) => value * 0.92), 11),
				parameters: 178,
			};
		case "mobilenet":
			return {
				logits: familyDense(
					input.map(
						(value, index) =>
							value - (input[(index + 1) % input.length] ?? 0) * 0.12,
					),
					12,
				),
				parameters: 82,
			};
		case "convnext":
			return { logits: residual3(input, 13), parameters: 238 };
		case "capsule": {
			const norm = Math.sqrt(
				input.reduce((sum, value) => sum + value * value, 0),
			) || 1;
			return {
				logits: familyDense(input.map((value) => value / norm), 14),
				parameters: 264,
			};
		}
		case "bilstm": {
			const forward = runRNN(input);
			const backward = runRNN(input.slice().reverse());
			return {
				logits: forward.map(
					(value, index) =>
						(value + backward[index]) * 0.5 +
						(value - backward[index]) * 0.18,
				),
				parameters: 128,
			};
		}
		case "transformer":
			return {
				logits: familyDense(input.slice().sort((a, b) => b - a), 17),
				parameters: 318,
			};
		case "encoder-transformer":
			return { logits: familyDense(input, 18), parameters: 302 };
		case "decoder-transformer":
			return {
				logits: familyDense(
					input.map((value, index) => index === 0 ? 0 : value),
					19,
				),
				parameters: 306,
			};
		case "encoder-decoder":
			return {
				logits: familyDense(
					input.map(
						(value, index) =>
							(value + (input[(index + 1) % input.length] ?? 0)) / 2,
					),
					20,
				),
				parameters: 376,
			};
		case "vit":
			return {
				logits: runAttention([input[0], input[2], input[1], input[3]]),
				parameters: 344,
			};
		case "swin":
			return {
				logits: runAttention([input[0], input[1], input[3], input[2]]),
				parameters: 286,
			};
		case "perceiver":
			return {
				logits: familyDense(input.map((value) => value * 0.5 + 0.25), 23),
				parameters: 358,
			};
		case "gcn":
			return {
				logits: familyDense(
					input.map(
						(value, index) =>
							(value + (input[(index + 1) % input.length] ?? 0)) / 2,
					),
					24,
				),
				parameters: 118,
			};
		case "gat": {
			const neighbor = input.map(
				(value, index) =>
					(
						value +
						input[(index + 1) % input.length] * 0.7 +
						input[(index + 3) % input.length] * 0.3
					) / 2,
			);
			return {
				logits: runAttention(neighbor).map(
					(value, index) => value + input[index] * 0.08,
				),
				parameters: 146,
			};
		}
		case "graphsage":
			return {
				logits: familyDense(
					input.map(
						(value, index) =>
							(value + (input[(index + 2) % input.length] ?? 0)) / 2,
					),
					26,
				),
				parameters: 124,
			};
		case "mpnn":
			return {
				logits: familyStack(
					input.map(
						(value, index) =>
							value + (input[(index + 1) % input.length] ?? 0) * 0.1,
					),
					27,
				),
				parameters: 172,
			};
		case "vae": {
			const latent = familyDense(input, 28);
			return {
				logits: familyDense(latent.map((value) => value * 0.7), 29),
				parameters: 224,
			};
		}
		case "gan": {
			const generated = familyDense(input, 30);
			return {
				logits: familyDense(
					input.map(
						(value, index) => value - generated[index] * 0.2,
					),
					31,
					false,
				),
				parameters: 246,
			};
		}
		case "diffusion": {
			let denoised = input.slice();
			for (let step = 0; step < 3; step += 1) {
				const previous = denoised;
				denoised = previous.map(
					(value, index) =>
						value * 0.82 +
						(previous[(index + 1) % previous.length] ?? 0) * 0.09,
				);
			}
			return { logits: familyDense(denoised, 32), parameters: 388 };
		}
		case "moe": {
			const expertA = familyDense(input, 33);
			const expertB = familyDense(input.slice().reverse(), 34);
			const gate = softmax([input[0], input[1], input[2] ?? 0]);
			return {
				logits: expertA.map(
					(value, index) =>
						value * gate[index] + expertB[index] * (1 - gate[index]),
				),
				parameters: 272,
			};
		}
		case "siamese": {
			const left = familyDense(input, 35);
			const right = familyDense(input.slice().reverse(), 36);
			return {
				logits: left.map(
					(value, index) => 1 - Math.abs(value - right[index]),
				),
				parameters: 154,
			};
		}
		case "nerf":
			return {
				logits: familyDense(
					input.map(
						(value, index) =>
							Math.sin(value * (index + 1) * Math.PI),
					),
					37,
				),
				parameters: 196,
			};
		case "dqn": {
			const qValues = familyDense(input, 38, false);
			const best = argmax(qValues);
			return {
				logits: qValues.map(
					(value, index) => value + (index === best ? 0.12 : 0),
				),
				parameters: 138,
			};
		}
		case "hypernetwork": {
			const scale = 0.7 + Math.abs(input[0] ?? 0) * 0.6;
			return {
				logits: familyDense(input.map((value) => value * scale), 39),
				parameters: 284,
			};
		}
		default:
			throw new Error(`Unknown neural architecture family: ${id}`);
	}
}

export function forwardArchitecture(id, input, options = {}) {
	if (!ARCHITECTURE_FAMILY_IDS.includes(id)) {
		throw new Error(`Unknown neural architecture family: ${id}`);
	}
	const { logits: baseLogits, parameters } = familyLogits(id, input, options);
	const bias = biasFor(id);
	const logits = baseLogits.map((value, index) => value + bias[index]);
	const probabilities = softmax(logits);
	const classIndex = argmax(probabilities);
	return Object.freeze({
		model: id,
		logits: Object.freeze(logits),
		probabilities: Object.freeze(probabilities),
		classIndex,
		confidence: probabilities[classIndex],
		parameters,
	});
}

function crossEntropy(id, options = {}, samples = DEMO_SAMPLES) {
	return samples.reduce(
		(sum, sample) =>
			sum -
			Math.log(
				Math.max(
					1e-6,
					forwardArchitecture(id, sample.input, options).probabilities[
						sample.label
					],
				),
			),
		0,
	) / Math.max(1, samples.length);
}

export function trainArchitecture(
	id,
	epochs = 3,
	reward = 1,
	options = {},
	samples = DEMO_SAMPLES,
) {
	const beforeLoss = crossEntropy(id, options, samples);
	for (let epoch = 0; epoch < Math.max(1, Math.trunc(epochs)); epoch += 1) {
		for (const sample of samples) {
			const prediction = forwardArchitecture(id, sample.input, options);
			const current = biasFor(id);
			const gradient = prediction.probabilities.map(
				(probability, index) =>
					probability - (index === sample.label ? 1 : 0),
			);
			learnedBiases.set(
				id,
				current.map(
					(value, index) =>
						value - 0.04 * Number(reward || 1) * gradient[index],
				),
			);
		}
	}
	const afterLoss = crossEntropy(id, options, samples);
	const accuracy =
		samples.filter(
			(sample) =>
				forwardArchitecture(id, sample.input, options).classIndex ===
				sample.label,
		).length / Math.max(1, samples.length);
	return Object.freeze({
		model: id,
		beforeLoss,
		afterLoss,
		epochs: Math.max(1, Math.trunc(epochs)),
		reward,
		accuracy,
	});
}

function stableHash(value) {
	let hash = 2166136261;
	for (let index = 0; index < value.length; index += 1) {
		hash = Math.imul(hash ^ value.charCodeAt(index), 16777619);
	}
	return (hash >>> 0).toString(16).padStart(8, "0");
}

function seeded(seed) {
	let value = Math.abs(Math.trunc(Number(seed) || 1));
	return () => {
		value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
		return value / 4294967296;
	};
}

export function createComposition(name, architectureIds) {
	const ids = architectureIds.map(String);
	for (const id of ids) {
		if (!ARCHITECTURE_FAMILY_IDS.includes(id)) {
			throw new Error(`Unknown neural architecture family: ${id}`);
		}
	}
	const nodes = ids.map((architectureId, index) => ({
		id: `node-${index + 1}-${architectureId}`,
		architectureId,
	}));
	const edges = nodes.slice(1).map((node, index) => ({
		id: `edge-${nodes[index].id}-${node.id}`,
		from: nodes[index].id,
		to: node.id,
		weight: 1,
		inhibition: false,
	}));
	return {
		id: `composition-${stableHash(`${name}|${ids.join(",")}`)}`,
		name,
		nodes,
		edges,
		state: {},
		depth: 1,
		lineage: [...ids],
	};
}

function influence(base, source, edge) {
	const sign = edge.inhibition ? -1 : 1;
	const scaled = adaptToStandardPort(source).map(
		(value) => value * edge.weight * sign,
	);
	return base.map((value, index) => value + scaled[index]);
}

export function runComposition(composition, input) {
	const outputs = {};
	const predictions = {};
	const state = { ...composition.state };
	for (const node of composition.nodes) {
		let nodeInput = adaptToStandardPort(input);
		for (const edge of composition.edges.filter((item) => item.to === node.id)) {
			if (outputs[edge.from]) {
				nodeInput = influence(nodeInput, outputs[edge.from], edge);
			}
			if (edge.sharedStateKey && state[edge.sharedStateKey]) {
				nodeInput = influence(
					nodeInput,
					state[edge.sharedStateKey],
					{ ...edge, weight: edge.weight * 0.5 },
				);
			}
		}
		const prediction = forwardArchitecture(node.architectureId, nodeInput);
		const output = adaptToStandardPort(prediction.probabilities);
		outputs[node.id] = output;
		predictions[node.id] = prediction;
		const stateKey = `${node.id}:state`;
		state[stateKey] = output.map(
			(value, index) =>
				(state[stateKey]?.[index] ?? 0) * 0.7 + value * 0.3,
		);
	}
	const terminal = composition.nodes.length
		? outputs[composition.nodes.at(-1).id]
		: adaptToStandardPort(input);
	return Object.freeze({
		compositionId: composition.id,
		outputs: Object.freeze(outputs),
		predictions: Object.freeze(predictions),
		state: Object.freeze(state),
		terminal: Object.freeze(terminal),
		steps: composition.nodes.length,
	});
}

export function generateEmergentComposition({
	seed = 1,
	nodeCount = 4,
	allowInhibition = true,
	architectureIds = ARCHITECTURE_FAMILY_IDS,
} = {}) {
	const random = seeded(seed);
	const ids = architectureIds.length ? architectureIds : ARCHITECTURE_FAMILY_IDS;
	const count = Math.max(2, Math.min(32, Math.trunc(nodeCount)));
	const nodes = Array.from({ length: count }, (_, index) => ({
		id: `generated-${seed}-${index + 1}`,
		architectureId: ids[Math.floor(random() * ids.length)] || "mlp",
	}));
	const edges = [];
	for (let index = 1; index < nodes.length; index += 1) {
		const source = nodes[Math.floor(random() * index)];
		edges.push({
			id: `edge-${source.id}-${nodes[index].id}`,
			from: source.id,
			to: nodes[index].id,
			weight: 0.25 + random() * 0.75,
			inhibition: Boolean(allowInhibition && random() > 0.82),
			sharedStateKey: `emergent-state-${Math.floor(random() * 3)}`,
		});
		if (index > 1 && random() > 0.45) {
			const second = nodes[Math.floor(random() * index)];
			edges.push({
				id: `edge-${second.id}-${nodes[index].id}-${edges.length}`,
				from: second.id,
				to: nodes[index].id,
				weight: -0.5 + random(),
				inhibition: Boolean(allowInhibition && random() > 0.88),
			});
		}
	}
	return {
		id: `composition-${stableHash(
			`generated|${seed}|${nodes.map((node) => node.architectureId).join(",")}`,
		)}`,
		name: `Emergent field ${seed}`,
		nodes,
		edges,
		state: {},
		depth: 1,
		lineage: ["generated", ...nodes.map((node) => node.architectureId)],
	};
}

export function mutateComposition(composition, seed) {
	const random = seeded(seed);
	const clone = structuredClone(composition);
	const operation = Math.floor(random() * 3);

	if (operation === 0) {
		const architectureId =
			ARCHITECTURE_FAMILY_IDS[
				Math.floor(random() * ARCHITECTURE_FAMILY_IDS.length)
			];
		const node = {
			id: `mutant-${seed}-${clone.nodes.length + 1}`,
			architectureId,
		};
		const target = clone.nodes[Math.floor(random() * clone.nodes.length)];
		clone.nodes.push(node);
		if (target) {
			clone.edges.push({
				id: `edge-${target.id}-${node.id}`,
				from: target.id,
				to: node.id,
				weight: 0.2 + random() * 0.8,
				inhibition: random() > 0.85,
			});
		}
		clone.lineage.push(architectureId);
	} else if (operation === 1 && clone.edges.length) {
		const edge = clone.edges[Math.floor(random() * clone.edges.length)];
		edge.weight = -1 + random() * 2;
		edge.inhibition = random() > 0.78;
	} else if (clone.nodes.length > 2) {
		const from = clone.nodes[Math.floor(random() * clone.nodes.length)];
		const to = clone.nodes[Math.floor(random() * clone.nodes.length)];
		if (from.id !== to.id) {
			clone.edges.push({
				id: `edge-${from.id}-${to.id}-${clone.edges.length}`,
				from: from.id,
				to: to.id,
				weight: 0.15 + random() * 0.85,
				inhibition: random() > 0.82,
			});
		}
	}
	clone.id = `composition-${stableHash(`${composition.id}|mutation|${seed}`)}`;
	clone.name = `${composition.name} · mutation ${seed}`;
	clone.lineage.push(`mutation:${seed}`);
	return clone;
}

export function createCapabilitySubstrate(capabilities, sharedState = {}) {
	return {
		capabilities: capabilities.map((capability) => ({ ...capability })),
		sharedState: { ...sharedState },
	};
}

export function createSemanticRoutes(capabilityIds) {
	return Object.fromEntries(
		capabilityIds.map((id, index) => [
			id,
			{
				channels: [index % 4, (index + 1) % 4],
				gain: 0.18,
			},
		]),
	);
}

export function applyModulation(substrate, run, routes = {}) {
	const globalSignal =
		run.terminal.reduce((sum, value) => sum + value, 0) /
		Math.max(1, run.terminal.length);
	return {
		capabilities: substrate.capabilities.map((capability) => {
			const route = routes[capability.id];
			const signal = route
				? route.channels.reduce(
					(sum, channel) => sum + (run.terminal[channel] ?? 0),
					0,
				) / Math.max(1, route.channels.length)
				: globalSignal;
			const gain = route?.gain ?? 0.15;
			return {
				...capability,
				activation: Math.max(
					0,
					Math.min(
						1,
						capability.activation * (0.75 + signal * 0.5) +
							signal * gain +
							(route?.bias ?? 0),
					),
				),
			};
		}),
		sharedState: { ...substrate.sharedState, ...run.state },
	};
}

export class NeuralArchitectureRuntime {
	constructor() {
		this.id = "neural-architecture-generator";
		this.address = Object.freeze({ gate: 43, line: 1, color: 1, tone: 1, base: 1 });
		this.metadata = Object.freeze({
			capabilities: Object.freeze([
				"neural.architecture.generate",
				"neural.architecture.run",
				"neural.architecture.mutate",
				"neural.architecture.train",
				"neural.capability.modulate",
			]),
			lineage: "neural-architecture-generator",
			familyCount: ARCHITECTURE_FAMILY_IDS.length,
			role: "capability-modulation-not-capability-ownership",
		});
		this.compositions = new Map();
	}

	async call(input = {}) {
		const operation = input.operation || input.op || "generate";
		if (operation === "generate") {
			const composition = generateEmergentComposition(input);
			this.compositions.set(composition.id, composition);
			return structuredClone(composition);
		}
		if (operation === "run") {
			const composition =
				typeof input.composition === "string"
					? this.compositions.get(input.composition)
					: input.composition;
			if (!composition) throw new Error("Unknown neural composition.");
			return runComposition(composition, input.input || [0, 0, 0, 0]);
		}
		if (operation === "mutate") {
			const source =
				typeof input.composition === "string"
					? this.compositions.get(input.composition)
					: input.composition;
			if (!source) throw new Error("Unknown neural composition.");
			const mutated = mutateComposition(source, input.seed || 1);
			this.compositions.set(mutated.id, mutated);
			return structuredClone(mutated);
		}
		if (operation === "train") {
			return trainArchitecture(
				String(input.family || "mlp"),
				input.epochs || 3,
				input.reward ?? 1,
				input.options || {},
				input.samples || DEMO_SAMPLES,
			);
		}
		if (operation === "modulate") {
			const composition =
				typeof input.composition === "string"
					? this.compositions.get(input.composition)
					: input.composition;
			if (!composition) throw new Error("Unknown neural composition.");
			const run = runComposition(composition, input.input || [0, 0, 0, 0]);
			const substrate = createCapabilitySubstrate(input.capabilities || []);
			const routes = createSemanticRoutes(
				substrate.capabilities.map((capability) => capability.id),
			);
			return Object.freeze({
				run,
				substrate: applyModulation(substrate, run, routes),
			});
		}
		throw new Error(`Unsupported NeuralArchitecture operation: ${operation}`);
	}

	snapshot() {
		return Object.freeze({
			id: this.id,
			families: ARCHITECTURE_FAMILY_IDS.length,
			compositions: this.compositions.size,
			metadata: this.metadata,
			// learned family bias vectors and full compositions stay owned here
		});
	}
}

export default new NeuralArchitectureRuntime();
