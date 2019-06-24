module NonLinearFunctions
    ( NonLinearFunction
    , DerivativeFunction
    , sigmoid
    , sigmoidDerivative
    ) where

type NonLinearFunction = Float -> Float
type DerivativeFunction = Float -> Float

sigmoid :: NonLinearFunction
sigmoid n = 1 / (1 + exp (-n))

sigmoidDerivative :: DerivativeFunction
sigmoidDerivative n = n * (1 - n)
