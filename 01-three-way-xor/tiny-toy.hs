import System.Environment (lookupEnv)
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
    , layer0 :: Layer
    , synapse0 :: Synapse
    , output :: Layer
    } deriving (Show)

getIterations :: Int -> Maybe String -> Int
getIterations defaultIterations envIterations =
    case envIterations of
        Nothing -> defaultIterations
        Just str ->
            case readMaybe str :: Maybe Int of
                Nothing -> defaultIterations
                Just n -> n

sigmoid :: Float -> Float
sigmoid n = 1 / (1 + exp (-n))

sigmoidDerivative :: Float -> Float
sigmoidDerivative n = n * (1 - n)

unflatten :: [a] -> [[a]]
unflatten = map (\n -> [n])

flatten :: [[a]] -> [a]
flatten = foldr (++) []

deepMap :: (a -> b) -> [[a]] -> [[b]]
deepMap fn = map $ map fn

dotProduct :: Vector -> Vector -> Float
dotProduct vector1 vector2 = sum $ zipWith (*) vector1 vector2

transpose :: Matrix -> Matrix
transpose [] = []
transpose ([]:_) = []
transpose matrix = (map head matrix) : (transpose $ map tail matrix)

matrixMultiply :: Matrix -> Matrix -> Matrix
matrixMultiply matrix =
    let transposedMatrix = transpose matrix
    in map (\row -> map (dotProduct row) transposedMatrix)

forwardPropagator :: (Float -> Float) -> Layer -> Synapse -> Layer
forwardPropagator nonLinearFunction layer synapse =
    deepMap nonLinearFunction $ matrixMultiply synapse layer

forwardPropagate :: Layer -> Synapse -> Layer
forwardPropagate = forwardPropagator sigmoid

calculateError :: Layer -> Layer -> Error
calculateError expected actual = unflatten $ zipWith (-) (flatten expected) (flatten actual)

deltaCalculator :: (Float -> Float) -> Layer -> Error -> Delta
deltaCalculator derivativeFunction layer err =
    let derivatives = deepMap derivativeFunction layer
    in unflatten $ zipWith (*) (flatten err) (flatten derivatives)

calculateDelta :: Layer -> Error -> Delta
calculateDelta = deltaCalculator sigmoidDerivative

calculateUpdate :: Layer -> Delta -> Update
calculateUpdate layer delta =
    let transposedLayer = transpose layer
    in matrixMultiply delta transposedLayer

updateSynapse :: Layer -> Synapse -> Delta -> Synapse
updateSynapse layer synapse delta =
    let update = calculateUpdate layer delta
    in zipWith (zipWith (+)) synapse update

trainOnce :: Network -> Network
trainOnce (Network expectedOutput layer0 synapse0 _) =
    let
        layer1 = forwardPropagate layer0 synapse0
        layer1Error = calculateError expectedOutput layer1
        layer1Delta = calculateDelta layer1 layer1Error
        updatedSynapse = updateSynapse layer0 synapse0 layer1Delta
    in
        Network expectedOutput layer0 updatedSynapse layer1

train :: Int -> Network -> Network
train 0 state = state
train n state = train (n - 1) (trainOnce state)

generateRandomSynapse :: Int -> StdGen -> Matrix
generateRandomSynapse width g = unflatten $ take width $ randomRs (-1.0, 1.0) g

main = do
    let x = [[0,0,1], [0,1,1], [1,0,1], [1,1,1]]
    let y =[[0], [0], [1], [1]]

    envIterations <- lookupEnv "ITERATIONS"
    let iterations = getIterations 100000 envIterations

    let inputWidth = length $ head x
    let g = mkStdGen 1337
    let randomSynapse = generateRandomSynapse inputWidth g

    let initialState = Network y x randomSynapse []
    let finalState = train iterations initialState

    print "Output after training:"
    print $ output finalState

-- E.g.
-- [[3.0174488e-3],[2.460981e-3],[0.9979919],[0.9975376]]
--
-- Time: 1.003s (built with ghc -O)
-- Time: 4.267s (runghc)
