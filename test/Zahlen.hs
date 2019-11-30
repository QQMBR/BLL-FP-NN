{-# LANGUAGE DataKinds, TypeApplications, RankNTypes, FlexibleContexts, 
             PartialTypeSignatures, TypeFamilies, TypeOperators, ScopedTypeVariables #-}

import qualified Control.Foldl as L
import qualified Numeric.LinearAlgebra.Static as H
import Numeric.LinearAlgebra.Data
import GHC.TypeNats
import Data.List
import Numeric.Backprop
import System.Random
import System.Random.Shuffle
import Networks
import Optimierung
import MNIST
import Tuple 
import Data.Proxy

main :: IO ()
main = testDigits

antwort :: KnownNat n => H.R n -> Int
antwort = maxIndex . H.extract 

normalizeInput :: (KnownNat n, KnownNat m) 
               => (H.R n, H.R m)
               -> (H.R n, H.R m)
normalizeInput (x, y) = (x / 255, y)

multilayerPerceptron :: forall i o. KN i o => Modell _ (H.R i) (H.R o)
multilayerPerceptron = sigmoidNoBias @30 <~ sigmoid 

threeLayer :: forall i o n. (KN i o, KnownNat n) => Proxy n -> Modell _ (H.R i) (H.R o)
threeLayer _ = sigmoid @n <~ sigmoid 

mP' :: (KN i o) => Modell _ (H.R i) (H.R o)
mP' = leakyReLU @30 <~ leakyReLU

emptyLn = putStr "\n"

randomOS :: (Fractional a, Random a) => a -> a -> IO a 
randomOS x y = (* y) . subtract x <$> randomIO 

trainableRBM s p0 = makeFoldAll (Minibatch (10, L2 (1.0, 0.000))) (rbmcd1 s) p0 

testDigits :: IO ()
testDigits = do
    let importImage l i = fmap normalizeInput <$> unsafeGetBoth l i
    let inDownloads = (++) "/home/julian/Downloads/"
    let lpath  = inDownloads "train-labels-idx1-ubyte"
    let ipath  = inDownloads "train-images-idx3-ubyte"
    let tlpath = inDownloads "t10k-labels-idx1-ubyte"
    let tipath = inDownloads "t10k-images-idx3-ubyte"

    test  <- importImage tlpath tipath
    train <- importImage lpath  ipath
    trainData <- concat <$> (mapM shuffleM $ replicate 30 train)

    p0 <- randomIO 

    let trained = L.fold (uberwachtesFold (Minibatch (10, SGD 1.5)) multilayerPerceptron 
                          se' p0) trainData

    emptyLn
    print $ (/ 100) . sum . map (\(x, l) -> 
        if ((== antwort l) . antwort . prediction trained $ x) 
        then 1 
        else 0) $ test
    
    print $ auswerten (antwort . evalBP2 multilayerPerceptron trained) 
        (==) (map (\(x, a) -> (x, antwort a)) test)
    
    where pred p x i l = (antwort l == x) && ((== x) . antwort . prediction p $ i)
          prediction p i = evalBP2 multilayerPerceptron p i


testAND :: IO ()
testAND = do 
    -- mische 100000 Samples 
    daten <- shuffleM . take 100000 . cycle $ samps              
    -- zufälliger Anfangswert, der resultierende Vektor von randomIO 
    -- ist standard normal verteilt
    p0 <- (* 0.1) <$> randomIO
    -- Netz wird trainiert und optimiert mit SGD und der Lernrate 0.1
    let trainiert = L.fold (uberwachtesFold (SGD 0.1) linear2 se p0) daten
    -- Hilfsfunktion für Ausgabe
    let disp = printLogicOpWith (evalBP2 linear2 trainiert) 
    
    emptyLn
    disp 0 0
    disp 1 0
    disp 0 1
    disp 1 1

    print $ auswerten (evalBP2 linear2 trainiert) (\x y -> round x == round y) daten
    
-- Die vier (Eingabe, Ziel) Paare, die gelernt werden
-- diese entsprechen dem logischen UND.
samps :: [(H.R 2, Double)]
samps = [(H.vec2 0 0, 0), (H.vec2 1 0, 0), (H.vec2 0 1, 0), (H.vec2 1 1, 1)]

printLogicOpWith :: (H.R 2 -> Double) -> Double -> Double -> IO ()
printLogicOpWith f a b = print $ disp (show . f $ H.vec2 a b) 
    where disp ans = "[" ++ show a ++ ", " ++ show b  ++ "]: " ++ ans
    
auswerten :: Ord b
          => (a -> b) 
          -> (b -> b -> Bool) 
          -> [(a, b)] 
          -> [Int] 
auswerten f pred xs = map (foldr sumCorrect 0) . groupBy sndEq . sortBy comp $ xs 
    where sndEq (_, b1) (_, b2) = b1 == b2 
          comp  (_, b1) (_, b2) = compare b1 b2
          sumCorrect x n        = if (eval x) then n+1 else n
          eval (a, b)           = pred (f a) b
