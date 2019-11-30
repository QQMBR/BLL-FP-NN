{-# LANGUAGE RankNTypes, TupleSections, ConstraintKinds, LambdaCase, ViewPatterns, 
             FlexibleContexts, DeriveGeneric, TypeOperators, PatternSynonyms #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module Tuple
(pattern (:&&), (:&) (..), snd', fst', mkT) where

import Numeric.LinearAlgebra.Static.Backprop
import Numeric.LinearAlgebra.Static.Vector
import Numeric.OneLiner
import Numeric.Backprop
import Lens.Micro hiding ((&))
import Data.List
import Control.Monad.Trans.State
import GHC.Generics (Generic)
import GHC.TypeNats
import System.Random
import qualified Data.Vector.Storable.Sized as SVS
import Data.Random.Normal

data a :& b = !a :& !b  
    deriving (Show, Generic)
infixr 2 :&

pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2) -> (x, y))
    where (:&&) = isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}

t1 :: Lens (a :& b) (a' :& b) a a'
t1 f (x :& y) = (:& y) <$> f x

t2 :: Lens (a :& b) (a :& b') b b'
t2 f (x :& y) = (x :&) <$> f y

fst' (a :& _) = a
snd' (_ :& b) = b

mkT = (:&)

instance (Num a, Num b) => Num (a :& b) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Fractional a, Fractional b) => Fractional (a :& b) where
    (/) = gDivide
    recip = gRecip
    fromRational = gFromRational

instance (Floating a, Floating b) => Floating (a :& b) where
    (**)  = gPower
    sqrt  = gSqrt
    acos  = gAcos
    asin  = gAsin
    atan  = gAtan
    cos   = gCos
    sin   = gSin
    tan   = gTan
    acosh = gAcosh
    asinh = gAsinh
    atanh = gAtanh
    cosh  = gCosh
    sinh  = gSinh
    tanh  = gTanh
    exp   = gExp
    log   = gLog
    pi    = gPi

instance (Random a, Random b) => Random (a :& b) where
    random g0 = (x :& y, g2)
      where
        (x, g1) = random g0
        (y, g2) = random g1
    randomR (x0 :& y0, x1 :& y1) g0 = (x :& y, g2)
      where
        (x, g1) = randomR (x0, x1) g0
        (y, g2) = randomR (y0, y1) g1

instance (Backprop a, Backprop b) => Backprop (a :& b)

uncurryT
    :: (Backprop a, Backprop b, Reifies z W)
    => (BVar z a -> BVar z b -> BVar z c)
    -> BVar z (a :& b)
    -> BVar z c
uncurryT f x = f (x ^^. t1) (x ^^. t2)

normalize :: KnownNat n => SVS.Vector n Double -> SVS.Vector n Double
normalize v = SVS.map (\x -> (x - meanv) / stdv) v
    where mean x = SVS.sum x / fromIntegral (SVS.length x) 
          std    = (** 0.5) . mean . SVS.map ((^2) . subtract (mean v))
          meanv  = mean v
          stdv   = std  v

{-
instance (KnownNat n, KnownNat m) => Random (L n m) where
    random = runState $ (vecL . normalize) <$> SVS.replicateM (state random)
    randomR (xs,ys) = runState . fmap vecL $ SVS.zipWithM (curry (state . randomR))
        (lVec xs) (lVec ys)

instance (KnownNat n) => Random (R n) where
    random = runState $ (vecR . normalize) <$> SVS.replicateM (state random)
    {-
        where norm mn v = SVS.map (\x -> (x - mn)/(std mn v)^0.5) v
              mean    v = SVS.sum v / fromIntegral (SVS.length v)
              std mn    = mean . SVS.map ((^2) . abs . subtract mn) 
              -}
        
    randomR (xs,ys) = runState . fmap vecR $ SVS.zipWithM (curry (state . randomR))
        (rVec xs) (rVec ys)
-}

instance (KnownNat n, KnownNat m) => Random (L n m) where
    random = runState $ vecL <$> SVS.replicateM (state normal)
    randomR (lo, hi) = undefined
    
instance (KnownNat n) => Random (R n) where
    random = runState $ vecR <$> SVS.replicateM (state normal)
    randomR (lo, hi) = undefined

