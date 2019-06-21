import System.Environment (lookupEnv)
import Utils
    ( Network(..)
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
    )


trainOnce :: Network -> Network
trainOnce (Network expectedOutput (layer0:_) (synapse0:synapse1:_)) =
    let
        layer1 = forwardPropagate sigmoid layer0 synapse0
        layer2 = forwardPropagate sigmoid layer1 synapse1

        layer2Error = calculateError expectedOutput layer2
        layer2Delta = calculateDelta sigmoidDerivative layer2 layer2Error

        layer1Error = calculateHiddenError layer2Delta synapse1
        layer1Delta = calculateDelta sigmoidDerivative layer1 layer1Error

        updatedSynapse1 = updateSynapse layer1 layer2Delta synapse1
        updatedSynapse0 = updateSynapse layer0 layer1Delta synapse0
    in
        Network expectedOutput [layer0, layer1, layer2] [updatedSynapse0, updatedSynapse1]

main = do
    envIterations <- lookupEnv "ITERATIONS"
    let iterations = getIterations 100000 envIterations

    let x = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    let y =[[0], [1], [1], [0]]

    let hiddenWidth = 10
    let widths = [getLayerWidth x, hiddenWidth, getLayerWidth y]
    let initialSynapses = generateRandomSynapses 1337 widths
    let initialState = Network y [x] initialSynapses

    let finalState = train trainOnce iterations initialState

    print "Output after training:"
    print $ last $ layers finalState
