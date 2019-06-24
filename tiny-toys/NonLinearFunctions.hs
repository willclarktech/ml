module NonLinearFunctions
    ( NonLinearFunction
    , DerivativeFunction
    , leakyRelu
    , leakyReluDerivative
    , relu
    , reluDerivative
    , sigmoid
    , sigmoidDerivative
    , softplus
    , softplusDerivative
    ) where

type NonLinearFunction = Float -> Float
type DerivativeFunction = Float -> Float

sigmoid :: NonLinearFunction
sigmoid n = 1 / (1 + exp (-n))

sigmoidDerivative :: DerivativeFunction
sigmoidDerivative n = n * (1 - n)

relu :: NonLinearFunction
relu n = max 0 n

reluDerivative :: DerivativeFunction
reluDerivative n
    | n <= 0 = 0
    | otherwise = n

leakyRelu :: NonLinearFunction
leakyRelu n
    | n <= 0 = 0.01 * n
    | otherwise = n

leakyReluDerivative :: DerivativeFunction
leakyReluDerivative n
    | n <= 0 = 0.01
    | otherwise = n

softplus :: NonLinearFunction
softplus n = log (1 + exp n)

softplusDerivative :: DerivativeFunction
softplusDerivative = sigmoid
