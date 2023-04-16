const WIDTH = 1100;
const HEIGHT = 500;
const STROKE_WEIGHT = 3;
const CROP_PADDING = (REPOS_PADDING = 2);
const COLOR = "#ffffff"

let english = ["moon","star","apple","cup","door","moustache","mountain","stairs","tornado","potato","table","chair", "rainbow", "light bulb", "bracelet"];
let level = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
let english_word = "";
let spanish = ["la luna","la estrella","la manzana","la taza","la puerta","el bigote","la montaña","las escalera","el tornado","la papa","la mesa","la silla", "el arco iris", "la bombilla", "la pulsera"];
let spanish_word = "";
let word_index = 0;
let mastery = 0;

let model;
let pieChart;
let clicked = false;
let mousePosition = [];
let solved;

// Coordinates of the current drawn stroke [[x1, x2, ..., xn], [y1, y2, ..., yn]]
let strokePixels = [[], []];

// Coordinates of all canvas strokes [[[x1, x2, ..., xn], [y1, y2, ..., yn]], [[x1, x2, ..., xn], [y1, y2, ..., yn]], ...]
let imageStrokes = [];

function inRange(n, from, to) {
	return n >= from && n < to;
}

function setup() {
	createCanvas(WIDTH, HEIGHT);
	strokeWeight(STROKE_WEIGHT);
	stroke("black");
	background(COLOR);
}

// from https://stackoverflow.com/questions/1431094/how-do-i-replace-a-character-at-a-particular-index-in-javascript
String.prototype.replaceAt = function(index, replacement) {
    return this.substring(0, index) + replacement + this.substring(index + replacement.length);
}

function removeLetters(v, word) { //v=level
	let l = word.length;
	let n = Math.round(l*(v/2)) //level 0: 0, level 1: 1/2, level 2: 1. n=number of letters to remove
	let u = [];
	for (let i=0; i<n; i++) {
		ind = Math.floor(Math.random()*l); //index to change
		if (u.some((e)=>e==ind)) {
			i--;
		} else {
			word = word.replaceAt(ind,"_");
			u.push(ind);
		}
	}
	return word;
}

function getNewWords() {
	word_index = Math.floor(Math.random()*12);
	english_word=english[word_index];
	spanish_word=spanish[word_index];
	english_word=removeLetters(level[word_index], english_word);
}

function modifyLevel(v) { // +1 if success /-1 if fail
	if (v<0 && level[word_index]>0) {
		level[word_index] -= 1;
	}
	if (v<0 && level[word_index]==3) {
		level[word_index] -= 1;
		mastery--;
	}
	
	if (v>0 && level[word_index]<2) {
		level[word_index] += 1;
	} else if (level[word_index]==2) {
		level[word_index] += 1;
		mastery++;
	}
}

function mouseDown() {
	clicked = true;
	mousePosition = [mouseX, mouseY];
}

function mouseMoved() {
	// Check whether mouse position is within canvas
	if (clicked && inRange(mouseX, 0, WIDTH) && inRange(mouseY, 0, HEIGHT)) {
		strokePixels[0].push(Math.floor(mouseX));
		strokePixels[1].push(Math.floor(mouseY));

		line(mouseX, mouseY, mousePosition[0], mousePosition[1]);
		mousePosition = [mouseX, mouseY];
	}
}

function mouseReleased() {
	if (strokePixels[0].length) {
		imageStrokes.push(strokePixels);
		strokePixels = [[], []];
	}
	clicked = false;
}

const loadModel = async () => {
	console.log("Model loading...");

	model = await tflite.loadTFLiteModel("./models/model.tflite");
	model.predict(tf.zeros([1, 28, 28, 1])); // warmup

	console.log(`Model loaded! (${LABELS.length} classes)`);
};

const preprocess = async (cb) => {
	const { min, max } = getBoundingBox();

	// Resize to 28x28 pixel & crop
	const imageBlob = await fetch("/transform", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		redirect: "follow",
		referrerPolicy: "no-referrer",
		body: JSON.stringify({
			strokes: repositionImage(),
			box: [min.x, min.y, max.x, max.y],
		}),	
	}).then((response) => response.blob());

	const img = new Image(28, 28);
	img.src = URL.createObjectURL(imageBlob);

	img.onload = () => {
		const tensor = tf.tidy(() =>
			tf.browser.fromPixels(img, 1).toFloat().expandDims(0)
		);
		cb(tensor);
	};
};

const drawPie = (top3) => {
	const probs = [];
	const labels = [];

	for (const pred of top3) {
		const prop = +pred.probability.toPrecision(2);
		probs.push(prop);
		labels.push(`${pred.className} (${prop})`);
	}

	const others = +(
		1 - probs.reduce((prev, prob) => prev + prob, 0)
	).toPrecision(2);
	probs.push(others);
	labels.push(`Others (${others})`);

	if (pieChart) pieChart.destroy();

	const ctx = document.getElementById("predictions").getContext("2d");
	pieChart = new Chart(ctx, {
		type: "pie",
		options: {
			plugins: {
				legend: {
					position: "bottom",
				},
				title: {
					display: true,
					text: "Top 3 Predictions",
				},
			},
		},
		data: {
			labels,
			datasets: [
				{
					label: "Top 3 predictions",
					data: probs,
					backgroundColor: [
						"rgb(255, 99, 132)",
						"rgb(54, 162, 235)",
						"rgb(255, 205, 86)",
						"rgb(97,96,96)",
					],
				},
			],
		},
	});
};

const getMinimumCoordinates = () => {
	let min_x = Number.MAX_SAFE_INTEGER;
	let min_y = Number.MAX_SAFE_INTEGER;

	for (const stroke of imageStrokes) {
		for (let i = 0; i < stroke[0].length; i++) {
			min_x = Math.min(min_x, stroke[0][i]);
			min_y = Math.min(min_y, stroke[1][i]);
		}
	}

	return [Math.max(0, min_x), Math.max(0, min_y)];
};

// Reposition image to top left corner
const repositionImage = () => {
    const [min_x, min_y] = getMinimumCoordinates();
    let newStroke = [[], []];
    let repositionedImageStrokes = [];
    for (const stroke of imageStrokes) {
        newStroke = [[], []];
        for (let i = 0; i < stroke[0].length; i++) {
            newStroke[0].push(stroke[0][i] - min_x + REPOS_PADDING);
            newStroke[1].push(stroke[1][i] - min_y + REPOS_PADDING);
        }
        repositionedImageStrokes.push(newStroke);
    }
    return repositionedImageStrokes;
};


const getBoundingBox = () => {
	const repositionedImage = repositionImage();

	const coords_x = [];
	const coords_y = [];

	for (const stroke of repositionedImage) {
		for (let i = 0; i < stroke[0].length; i++) {
			coords_x.push(stroke[0][i]);
			coords_y.push(stroke[1][i]);
		}
	}

	const x_min = Math.min(...coords_x);
	const x_max = Math.max(...coords_x);
	const y_min = Math.min(...coords_y);
	const y_max = Math.max(...coords_y);

	// New width & height of cropped image
	const width = Math.max(...coords_x) - Math.min(...coords_x);
	const height = Math.max(...coords_y) - Math.min(...coords_y);

	const coords_min = {
		x: Math.max(0, x_min - CROP_PADDING), // Link Kante anlegen
		y: Math.max(0, y_min - CROP_PADDING), // Obere Kante anlegen
	};
	let coords_max;

	if (width > height)
		// Left + right edge as boundary
		coords_max = {
			x: Math.min(WIDTH, x_max + CROP_PADDING), // Right edge
			y: Math.max(0, y_min + CROP_PADDING) + width, // Lower edge
		};
	// Upper + lower edge as boundary
	else
		coords_max = {
			x: Math.max(0, x_min + CROP_PADDING) + height, // Right edge
			y: Math.min(HEIGHT, y_max + CROP_PADDING), // Lower edge
		};

	return {
		min: coords_min,
		max: coords_max,
	};
};

function tryComplete(b) {
	if (b) {
		showNextWordButton();
	}else {
		hideNextWordButton();
	}
}

function showNextWordButton() {
	const $rightArrow = document.getElementById("rightArrow");
	$rightArrow.classList.remove("d-none");
}

function hideNextWordButton() {
	const $rightArrow = document.getElementById("rightArrow");
	$rightArrow.classList.add("d-none");
}

function nextWord() {
	clearCanvas();
	modifyLevel(1);
	hideNextWordButton();
	getNewWords();
	setWords();
}

function skip() {
	modifyLevel(-1);
	clearCanvas();
	getNewWords();
	setWords();
}

function hint() {
	if (level[word_index]>0) {
		modifyLevel(-1);
		english_word=english[word_index];
		english_word=removeLetters(level[word_index], english_word);
		setWords();
	} 
}

const predict = async () => {
	if (!imageStrokes.length) return;
	if (!LABELS.length) throw new Error("No labels found!");

	preprocess((tensor) => {
		const predictions = model.predict(tensor).dataSync();

		const top3 = Array.from(predictions)
			.map((p, i) => ({
				probability: p,
				className: LABELS[i],
				index: i,
			}))
			.sort((a, b) => b.probability - a.probability)
			.slice(0, 3);
		
		solved = top3.some((e)=>e.className==english[word_index]); 
		tryComplete(solved);
		setGuess(solved ? english[word_index] : top3[0].className);

		drawPie(top3);
		//console.log(level);
	});
};

const clearCanvas = () => {
	clear();
	if (pieChart) pieChart.destroy();
	background(COLOR);
	imageStrokes = [];
	strokePixels = [[], []];
};

const setGuess = (word) => {
	const $guess = document.getElementById("guess");
	$guess.innerHTML = word;
}

const setWords = () => {
	const $knownword = document.getElementById("knownword");
	$knownword.innerHTML = english_word;
	const $unknownword = document.getElementById("unknownword");
	$unknownword.innerHTML = spanish_word;
	const $mastery = document.getElementById("mastery");
	$mastery.innerHTML = "Mastered: " + mastery + "/15";
}

const renderCanvas = () => {
    for (const stroke of imageStrokes) {
        for (let i = 0; i < stroke[0].length - 1; i++) {
            line(stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]);
        }
    }
    console.log("Canvas rendered");
};


const undoStroke = () => {
    imageStrokes.pop();
    clear();
	background(COLOR);
    renderCanvas();
}

window.onload = () => {
	getNewWords();
	//const $submit = document.getElementById("predict");
	const $clear = document.getElementById("clear");
	const $canvas = document.getElementById("defaultCanvas0");
	const $hint = document.getElementById("hint");
	const $skip = document.getElementById("skip");
	const $rightArrow = document.getElementById("rightArrow");
	const $undo = document.getElementById("undo");
	setWords();
	

	loadModel();
	$canvas.addEventListener("mousedown", (e) => mouseDown(e));
	$canvas.addEventListener("mouseup", () => {
		predict($canvas);
	});
	$canvas.addEventListener("mousemove", (e) => mouseMoved(e));

	$hint.addEventListener("click", () => {hint()});
	$skip.addEventListener("click", () => {skip()});
	//$submit.addEventListener("click", () => predict($canvas));
	$clear.addEventListener("click", clearCanvas);
	$rightArrow.addEventListener("click", nextWord);
	$undo.addEventListener("click", () => {
		undoStroke();
		predict($canvas);
	});
};

