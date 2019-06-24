import System.Environment (lookupEnv)
import NonLinearFunctions
    ( sigmoid
    , sigmoidDerivative
    )
import NeuralNetwork
    ( Network (..)
    , generateRandomSynapses
    , getIterations
    , getLayerWidth
    , train
    )

main = do
    envIterations <- lookupEnv "ITERATIONS"
    let iterations = getIterations 100000 envIterations

    let x = [[0,0,1], [0,1,1], [1,0,1], [1,1,1]]
    let y =[[0], [0], [1], [1]]

    let widths = [getLayerWidth x, getLayerWidth y]
    let initialSynapses = generateRandomSynapses 1337 widths
    let initialState = Network y [x] initialSynapses

    let finalState = train sigmoid sigmoidDerivative iterations initialState

    print "Output after training:"
    print $ last $ layers finalState

-- E.g.
-- [[3.0174488e-3],[2.460981e-3],[0.9979919],[0.9975376]]
--
-- Time: 1.003s (built with ghc -O)
-- Time: 4.267s (runghc)
