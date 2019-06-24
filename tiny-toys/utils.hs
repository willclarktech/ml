module Utils where

import Data.List (foldl')
import System.Random (mkStdGen, randomRs)
import Text.Read (readMaybe)

type NonLinearFunction = Float -> Float
type DerivativeFunction = Float -> Float

type Vector = [Float]

type Matrix = [[Float]]
type Layer = Matrix
type Synapse = Matrix
type Error = Matrix
type Delta = Matrix
type Update = Matrix

data Network = Network
    { expectedOutput :: Layer
    , layers :: [Layer]
    , synapses :: [Synapse]
    } deriving (Show)


getIterations :: Int -> Maybe String -> Int
getIterations defaultIterations Nothing = defaultIterations
getIterations defaultIterations (Just str) =
    case readMaybe str :: Maybe Int of
        Nothing -> defaultIterations
        Just n -> n

getLayerWidth :: Layer -> Int
getLayerWidth = length . head

mkRandomStream :: Int -> [Float]
mkRandomStream seed =
    let g = mkStdGen seed
    in randomRs (-1.0, 1.0) g

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk size arr
    | size > 0 = take size arr : chunk size (drop size arr)
    | otherwise = error "Cannot chunk into chunks of size 0 or less"

generateSynapses :: [Float] -> [Int] -> [Synapse]
generateSynapses valueStream [] = []
generateSynapses valueStream (_:[]) = []
generateSynapses valueStream (inputWidth:outputWidth:widths) =
    let
        numValues = inputWidth * outputWidth
        valuesToUse = take numValues valueStream
        unusedValues = drop numValues valueStream
        synapse = chunk outputWidth valuesToUse
    in synapse : (generateSynapses unusedValues (outputWidth:widths))

generateRandomSynapses :: Int -> [Int] -> [Synapse]
generateRandomSynapses seed widths =
    let randomStream = mkRandomStream seed
    in generateSynapses randomStream widths

sigmoid :: NonLinearFunction
sigmoid n = 1 / (1 + exp (-n))

sigmoidDerivative :: DerivativeFunction
sigmoidDerivative n = n * (1 - n)

flatten :: [[a]] -> [a]
flatten = foldr (++) []

deepMap :: (a -> b) -> [[a]] -> [[b]]
deepMap = map . map

dotProduct :: Vector -> Vector -> Float
dotProduct vector1 = sum . (zipWith (*) vector1)

transpose :: Matrix -> Matrix
transpose [] = []
transpose ([]:_) = []
transpose matrix = (map head matrix) : (transpose $ map tail matrix)

matrixMultiply :: Matrix -> Matrix -> Matrix
matrixMultiply matrix1 matrix2 =
    let transposedMatrix = transpose matrix2
    in map (\row -> map (dotProduct row) transposedMatrix) matrix1

propagateLayer :: NonLinearFunction -> Layer -> Synapse -> Layer
propagateLayer nonLinearFunction layer =
    (deepMap nonLinearFunction) . (matrixMultiply layer)


propagateNetwork :: NonLinearFunction -> Layer -> [Synapse] -> [Layer]
propagateNetwork nonLinearFunction inputLayer synapses =
    foldl' (\layers synapse -> layers ++ [propagateLayer nonLinearFunction (last layers) synapse]) [inputLayer] synapses

calculateOutputError :: Layer -> Layer -> Error
calculateOutputError expected actual =
    map (\(e, a) -> zipWith (-) e a) $ zip expected actual

calculateHiddenError :: Delta -> Synapse -> Error
calculateHiddenError delta synapse = matrixMultiply delta $ transpose synapse

calculateDelta :: DerivativeFunction -> Layer -> Error -> Delta
calculateDelta derivativeFunction layer err =
    let
        chunkSize = getLayerWidth layer
        derivatives = deepMap derivativeFunction layer
    in chunk chunkSize $ zipWith (*) (flatten err) (flatten derivatives)

backpropagateLayer :: DerivativeFunction -> (Layer, Synapse) -> [Delta] -> [Delta]
backpropagateLayer derivativeFunction (layer, synapse) deltas =
    let
        error = calculateHiddenError (head deltas) synapse
        delta = calculateDelta derivativeFunction layer error
    in delta:deltas

backpropagateNetwork :: DerivativeFunction -> Layer -> [Layer] -> [Synapse] -> [Delta]
backpropagateNetwork derivativeFunction expectedOutput layers synapses =
    let
        outputLayer = last layers
        outputError = calculateOutputError expectedOutput outputLayer
        outputDelta = calculateDelta derivativeFunction outputLayer outputError
        zipped = zip layers synapses
    in
        foldr (backpropagateLayer derivativeFunction) [outputDelta] zipped

calculateUpdate :: Layer -> Delta -> Update
calculateUpdate layer = matrixMultiply (transpose layer)

updateSynapse :: Layer -> Delta -> Synapse -> Synapse
updateSynapse layer delta = zipWith (zipWith (+)) $ calculateUpdate layer delta

updateSynapses :: [Synapse] -> [Layer] -> [Delta] -> [Synapse]
updateSynapses synapses layers deltas =
    let zipped = zip3 synapses layers deltas
    in map (\(synapse, layer, delta) -> updateSynapse layer delta synapse) zipped

trainOnce :: NonLinearFunction -> DerivativeFunction -> Network -> Network
trainOnce nonLinearFn derivativeFn (Network expectedOutput (inputLayer:_) synapses) =
    let
        layers = propagateNetwork nonLinearFn inputLayer synapses
        deltas = backpropagateNetwork derivativeFn expectedOutput (tail layers) (tail synapses)
        newSynapses = updateSynapses synapses layers deltas

    in Network expectedOutput layers newSynapses

train :: NonLinearFunction -> DerivativeFunction -> Int -> Network -> Network
train _ _ 0 state = state
train nonLinearFn derivativeFn n state =
    let trainedOnce = trainOnce nonLinearFn derivativeFn state
    in train nonLinearFn derivativeFn (n - 1) trainedOnce
