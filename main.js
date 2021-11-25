const tf = require('@tensorflow/tfjs-node');
const fs = require('fs')

// util function to normalize a value between a given range.
function normalize(value, min, max) {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value - min) / (max - min);
}

// data can be loaded from URLs or local file paths when running in Node.js.
const TRAIN_DATA_PATH = 'file://G:/Code/js/kaggle/titanic/data/train.csv';
const TEST_DATA_PATH = 'file://G:/Code/js/kaggle/titanic/data/test.csv';

const SURVIVED_CLASSES = 2;
const TRAINING_DATA_LENGTH = 800;
const TEST_DATA_LENGTH = 91;

// Converts a row from the CSV into features and labels.
// Each feature field is normalized within training data constants
const csvTransform =
    ({xs, ys}) => {
      Embarked = -1
      if (xs.Embarked == 'S') {Embarked = 0} else if (xs.Embarked == 'C') {Embarked = 1} else {Embarked = 2}
      const values = [
        normalize(xs.Pclass, 1, 3),
        normalize(xs.Sex == 'male' ? 1 : 0, 0, 1),
        normalize(xs.age, -1, 80),
        normalize(xs.SibSp, 0, 8),
        normalize(xs.Parch, 0, 6),
        normalize(xs.Fare, 0, 512.3292),
        normalize(Embarked, 0, 2),
      ];
      if (Number.isNaN(values[2])) {
        values[2] = normalize(-1, -1, 80)
      }
      return {xs: values, ys: ys.Survived};
    }

const trainingData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {Survived: {isLabel: true}}})
        .map(csvTransform)
    //    .shuffle(TRAINING_DATA_LENGTH)
        .batch(100);

// Load all training data in one batch to use for evaluation
const trainingValidationData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {Survived: {isLabel: true}}})
        .map(csvTransform)
        .batch(TRAINING_DATA_LENGTH);

// Load all test data in one batch to use for evaluation
const testValidationData =
    tf.data.csv(TEST_DATA_PATH, {columnConfigs: {Survived: {isLabel: true}}})
        .map(csvTransform)
        .batch(TEST_DATA_LENGTH);

const model = tf.sequential();
model.add(tf.layers.dense({units: 250, activation: 'relu', inputShape: [7]}));
model.add(tf.layers.dense({units: 175, activation: 'relu'}));
model.add(tf.layers.dense({units: 150, activation: 'relu'}));
model.add(tf.layers.dense({units: SURVIVED_CLASSES, activation: 'softmax'}));

model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

// Returns pitch class evaluation percentages for training data
// with an option to include test data
async function evaluate(useTestData) {
  let results = {};
  await trainingValidationData.forEachAsync(pitchTypeBatch => {
    const values = model.predict(pitchTypeBatch.xs).dataSync();
    const classSize = TRAINING_DATA_LENGTH / SURVIVED_CLASSES;
    for (let i = 0; i < SURVIVED_CLASSES; i++) {
      results[pitchFromClassNum(i)] = {
        training: calcPitchClassEval(i, classSize, values)
      };
    }
  });

  if (useTestData) {
    await testValidationData.forEachAsync(pitchTypeBatch => {
      const values = model.predict(pitchTypeBatch.xs).dataSync();
      const classSize = TEST_DATA_LENGTH / SURVIVED_CLASSES;
      for (let i = 0; i < SURVIVED_CLASSES; i++) {
        results[pitchFromClassNum(i)].validation =
            calcPitchClassEval(i, classSize, values);
      }
    });
  }
  return results;
}

async function predictSample(sample) {
  let result = model.predict(tf.tensor(sample, [1,sample.length])).arraySync();
  var maxValue = 0;
  var predictedPitch = 7;
  for (var i = 0; i < SURVIVED_CLASSES; i++) {
    if (result[0][i] > maxValue) {
      predictedPitch = i;
      maxValue = result[0][i];
    }
  }
  return predictedPitch
}

// Determines accuracy evaluation for a given pitch class by index
function calcPitchClassEval(pitchIndex, classSize, values) {
  // Output has 7 different class values for each pitch, offset based on
  // which pitch class (ordered by i)
  let index = (pitchIndex * classSize * SURVIVED_CLASSES) + pitchIndex;
  let total = 0;
  for (let i = 0; i < classSize; i++) {
    total += values[index];
    index += SURVIVED_CLASSES;
  }
  return total / classSize;
}

// Returns the string value for Baseball pitch labels
function pitchFromClassNum(classNum) {
  switch (classNum) {
    case 0:
      return 'Dead';
    case 1:
      return 'Alive';
    default:
      return 'Unknown';
  }
}

async function run() {
  await model.fitDataset(trainingData, {epochs: 200});
  console.log('accuracyPerClass', await evaluate(true));

  output = "PassengerId,Survived\n"
  input = fs.readFileSync("data/fintest.csv", "utf8").split("\n")
  for (var i = 1; i < input.length-1; i++) {
    split = input[i].split(",")
    Embarked = -1
    if (split[11] == 'S') {Embarked = 0} else if (split[11] == 'C') {Embarked = 1} else {Embarked = 2}
    const values = [
      normalize(split[1], 1, 3),
      normalize(split[4] == 'male' ? 1 : 0, 0, 1),
      normalize(split[5], -1, 80),
      normalize(split[6], 0, 8),
      normalize(split[7], 0, 6),
      normalize(split[9], 0, 512.3292),
      normalize(Embarked, 0, 2),
    ];
    if (Number.isNaN(values[2])) {
      values[2] = normalize(-1, -1, 80)
    }
    value = await predictSample(values)

    output += split[0]+","+value+"\n"
  }
  fs.writeFileSync("fintestoutput.csv", output)
}

run()
