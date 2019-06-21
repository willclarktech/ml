module Utils
    ( Network (..)
    , calculateDelta
    , calculateError
    , calculateHiddenError
    , forwardPropagate
    , generateRandomSynapses
    , getIterations
    , getLayerWidth
    , sigmoid
    , sigmoidDerivative
    , train
    , updateSynapse
    ) where

import System.Random (mkStdGen, randomRs, StdGen)
import Text.Read (readMaybe)


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

sigmoid :: Float -> Float
sigmoid n = 1 / (1 + exp (-n))

sigmoidDerivative :: Float -> Float
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

forwardPropagate :: (Float -> Float) -> Layer -> Synapse -> Layer
forwardPropagate nonLinearFunction layer =
    (deepMap nonLinearFunction) . (matrixMultiply layer)

calculateError :: Layer -> Layer -> Error
calculateError expected actual =
    let chunkSize = getLayerWidth expected
    in chunk chunkSize $ zipWith (-) (flatten expected) (flatten actual)

calculateHiddenError :: Delta -> Synapse -> Error
calculateHiddenError delta synapse = matrixMultiply delta $ transpose synapse

calculateDelta :: (Float -> Float) -> Layer -> Error -> Delta
calculateDelta derivativeFunction layer err =
    let
        chunkSize = getLayerWidth layer
        derivatives = deepMap derivativeFunction layer
    in chunk chunkSize $ zipWith (*) (flatten err) (flatten derivatives)

calculateUpdate :: Layer -> Delta -> Update
calculateUpdate layer =
    let transposedLayer = transpose layer
    in matrixMultiply transposedLayer

updateSynapse :: Layer -> Synapse -> Delta -> Synapse
updateSynapse layer delta =
    let update = calculateUpdate layer delta
    in zipWith (zipWith (+)) update

train :: (Network -> Network) -> Int -> Network -> Network
train _ 0 state = state
train trainOnce n state = train trainOnce (n - 1) $ trainOnce state
