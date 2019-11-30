{-# LANGUAGE RankNTypes,
             ScopedTypeVariables,
             ViewPatterns,
             TypeFamilies,
             TypeOperators,
             RankNTypes,
             MultiWayIf,
             DataKinds,
             LambdaCase,
             ConstraintKinds, 
             FlexibleContexts #-}

module Networks where 

import Optimierung
import Tuple
import Numeric.LinearAlgebra.Static.Backprop
import Numeric.Backprop
import GHC.TypeNats
import qualified Numeric.LinearAlgebra.Static as H
import qualified Control.Foldl as L
import qualified Data.Vector.Storable as V

type KN i o = (KnownNat i, KnownNat o)
type Fehler a b = forall z. Reifies z W => BVar z a -> BVar z a -> BVar z b
type Res p a b c = forall z. Reifies z W => BVar z (a :& b) -> BVar z p -> BVar z c

type Modell p a b = 
       forall z. Reifies z W
    => BVar z p
    -> BVar z a
    -> BVar z b
    
withScalarActivation :: (Reifies z W, KnownNat i) 
                     => (BVar z Double -> BVar z Double) 
                     -> BVar z (R i :& Double) 
                     -> BVar z (R i)
                     -> BVar z Double
withScalarActivation f (w :&& b) v = f $ w <.> v + b 

withActivation :: (Reifies z W, KN i o)
               => (BVar z (R o) -> BVar z (R o))
               -> BVar z (L o i :& R o)
               -> BVar z (R i)
               -> BVar z (R o)
withActivation f (w :&& b) v = f $ w #> v + b

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

linear :: KnownNat i => Modell (R i :& Double) (R i) Double
linear = withScalarActivation id

linear2 :: KnownNat i => Modell (R i :& Double) (R i) Double
linear2 (w :&& b) x = w <.> x + b

sigmoid :: KN i o => Modell (L o i :& R o) (R i) (R o) 
sigmoid = withActivation logistic

--todo: dvmap isn't actually made differentiable, manually make Op
leakyReLU :: KN i o => Modell (L o i :& R o) (R i) (R o)
leakyReLU = withActivation $ dvmap $ \x -> if (x < 0) then 0.01*x
                                                      else x

sigmoidNoBias :: KN i o => Modell (L o i) (R i) (R o)
sigmoidNoBias w v = logistic $ w #> v

sigmoid' :: KnownNat i => Modell (R i :& Double) (R i) Double
sigmoid' = withScalarActivation logistic  

se :: Fehler Double Double
se x a = (x - a)^2

se' :: (KnownNat n, 1 <= n) => Fehler (R n) Double
se' a e = (d <.> d) / 2
    where d = e - a

crossentropy' :: (KnownNat n, 1 <= n) => Fehler (R n) Double
crossentropy' a y = vsum . nanToNum . negate $ entropy y a + entropy (1-y) (1-a)
    where entropy x y = x * log y

crossentropyUnsafe' :: (KnownNat n, 1 <= n) => Fehler (R n) Double
crossentropyUnsafe' a y = vsum . negate $ entropy y a + entropy (1-y) (1-a)
    where entropy x y = x * log y
    
crossentropy :: (KnownNat n, 1 <= n) => R n -> R n -> Double
crossentropy a y = (H.konst 1) H.<.> (entropy (-y) a - entropy (1-y) (1-a))
    where entropy x y = x * log y
    
logisticSample :: KN i o => (L o i :& R o) -> Int -> R i -> R o 
logisticSample (w :& b) s v = H.zipWithVector (\x p -> if x < p then 1 else 0) r probs
    where probs = logistic $ w H.#> v + b
          r     = H.randomVector s H.Uniform

nonModelLogistic :: KN i o => (L o i :& R o) -> R i -> R o
nonModelLogistic (w :& b) v = logistic $ w H.#> v + b 

vsum :: (Reifies z W, KnownNat n) => BVar z (R n) -> BVar z Double
vsum x = (konst 1) <.> x

rbmcd1 :: KN d h => Int -> (L d h :& R d :& R h) -> R d -> (L d h :& R d :& R h)
rbmcd1 s (w :& b :& c) v1 = (H.outer v1 h1 - H.outer v2 h2) :& (v1 - v2) :& (h1 - h2)
    where h1 = (logisticSample (H.tr w :& c) s v1)
          v2 = (nonModelLogistic (w :& b) h1)
          h2 = (nonModelLogistic (H.tr w :& c) v2)

gradientStep :: (Backprop p, Backprop b, Backprop c)
             => Modell p a b  -- ^ Modell zum Trainieren 
             -> Fehler b c    -- ^ Fehlerfunktion
             -> p             -- ^ Parameter am Anfang des Schrittes
             -> (a, b)        -- ^ (Eingabedatum, erwartete Antwort) 
             -> p             -- ^ Gradient des Parameters
gradientStep mod err param (vec, trg) = gradBP (\p -> err (mod p (auto vec)) (auto trg)) param

makeFoldAll :: Optimierung d p 
         => d p
         -> (p -> a -> p)
         -> p
         -> L.Fold a p
makeFoldAll d f p0 = L.Fold (\x a -> regel d x (f (aus x) a)) (start p0) aus

uberwachtesFold :: (Optimierung d p, Backprop p, Backprop b, Backprop c)
                => d p 
                -> Modell p a b
                -> Fehler b c 
                -> p
                -> L.Fold (a, b) p
uberwachtesFold op mod err p0 = makeFoldAll op (gradientStep mod err) p0

isVecNaN :: KnownNat n => R n -> Bool
isVecNaN = V.foldl (\acc x -> acc || isNaN x) False . H.extract 

nanToNum :: (KnownNat n, Reifies z W) => BVar z (R n) -> BVar z (R n)
nanToNum = vmap bvToNum

bvToNum :: Reifies z W => BVar z Double -> BVar z Double
bvToNum = liftOp1 . op1 $ \x -> ((if
                            | isNaN x      -> 0.0
                            | isInfinite x -> if x < 0 then -1.7976931348623157e+308 
                                                       else  1.7976931348623157e+308
                            | otherwise    -> x), id)
                                            

supervised :: (Backprop a, Backprop p, Backprop b, Backprop c)
           => Fehler b c 
           -> Modell p a b
           -> Res p a b c
supervised err mod (a :&& y) ps = err (mod ps a) y

class Optimizable m where
    type Params m
    type In     m
    gradient :: m -> Params m -> In m -> Params m

makeFoldOptimizable :: (Optimizable m, Optimierung d (Params m)) 
                    => m
                    -> d (Params m)
                    -> Params m
                    -> L.Fold (In m) (Params m)
makeFoldOptimizable m d p0 = L.Fold (\x a -> regel d x (step (aus x) a)) (start p0) aus
    where step = gradient m

(<~) :: (Backprop p, Backprop q) 
     => Modell  p       b c
     -> Modell      q  a b
     -> Modell (p :& q) a c
(f <~ g) (p :&& q) = f p . g q 

infixr 8 <~

