module Utils where

import System.Random (mkStdGen, randomRs)


mkRandomStream :: Int -> [Float]
mkRandomStream seed =
    let g = mkStdGen seed
    in randomRs (-1.0, 1.0) g

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk size arr
    | size > 0 = take size arr : chunk size (drop size arr)
    | otherwise = error "Cannot chunk into chunks of size 0 or less"

flatten :: [[a]] -> [a]
flatten = foldr (++) []

deepMap :: (a -> b) -> [[a]] -> [[b]]
deepMap = map . map
